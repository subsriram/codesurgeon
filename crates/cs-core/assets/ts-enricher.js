#!/usr/bin/env node
'use strict';

// codesurgeon TypeScript enricher shim
// Spawned by codesurgeon at index time to resolve symbol types via the TS compiler API.
//
// Usage: node ts-enricher.js <workspace_root>
// Output: NDJSON to stdout, one record per enriched symbol:
//   { "fqn": "<rel_path>::[Container::]<name>", "resolved_type": "<type>", "line": N }
//
// Exit 0 always — non-zero is reserved for hard failures the caller should warn on.
// Graceful skips (no tsconfig, no typescript package) exit 0 with a stderr message.

const path = require('path');
const fs = require('fs');

const workspaceRoot = process.argv[2];
if (!workspaceRoot) {
  process.stderr.write('Usage: ts-enricher.js <workspace_root>\n');
  process.exit(1);
}

// ── Load TypeScript ───────────────────────────────────────────────────────────
// Try workspace-local typescript first, then fall back to any globally resolvable copy.

let ts;
const localTs = path.join(workspaceRoot, 'node_modules', 'typescript');
try {
  ts = require(localTs);
} catch (_) {
  try {
    ts = require('typescript');
  } catch (_2) {
    process.stderr.write(
      'TypeScript not found in ' + localTs + ' or globally — skipping ts-enrichment\n'
    );
    process.exit(0);
  }
}

// ── Load tsconfig.json ────────────────────────────────────────────────────────

const tsconfigPath = path.join(workspaceRoot, 'tsconfig.json');
if (!fs.existsSync(tsconfigPath)) {
  process.stderr.write('No tsconfig.json found in workspace root — skipping ts-enrichment\n');
  process.exit(0);
}

const configResult = ts.readConfigFile(tsconfigPath, ts.sys.readFile);
if (configResult.error) {
  process.stderr.write(
    'tsconfig.json read error: ' +
    ts.flattenDiagnosticMessageText(configResult.error.messageText, '\n') + '\n'
  );
  process.exit(0);
}

const parsedConfig = ts.parseJsonConfigFileContent(
  configResult.config,
  ts.sys,
  workspaceRoot
);

// ── Create program ────────────────────────────────────────────────────────────

const compilerOptions = Object.assign({}, parsedConfig.options, {
  allowJs: true,
  noEmit: true,
  skipLibCheck: true,
  // Disable strict null so we get concrete types rather than T | undefined everywhere.
  strictNullChecks: false,
});

const program = ts.createProgram({
  rootNames: parsedConfig.fileNames,
  options: compilerOptions,
});

const checker = program.getTypeChecker();

// ── Helpers ───────────────────────────────────────────────────────────────────

function relPath(absPath) {
  return path.relative(workspaceRoot, absPath).replace(/\\/g, '/');
}

const MAX_TYPE_LEN = 120;

function typeStr(type) {
  try {
    let s = checker.typeToString(
      type,
      undefined,
      ts.TypeFormatFlags
        ? (ts.TypeFormatFlags.NoTruncation | ts.TypeFormatFlags.WriteArrayAsGenericType)
        : 0
    );
    return s.length > MAX_TYPE_LEN ? s.slice(0, MAX_TYPE_LEN) + '…' : s;
  } catch (_) {
    return '';
  }
}

// Types that add no value as resolved_type annotations.
const SKIP_TYPES = new Set([
  'any', 'unknown', 'void', 'never', 'undefined', 'null', '{}', 'object',
]);

function isSkippable(s) {
  return !s || SKIP_TYPES.has(s);
}

/**
 * Walk up the AST to find the nearest enclosing class/interface/namespace name.
 * Returns undefined for top-level declarations.
 */
function containerName(node) {
  let p = node.parent;
  while (p) {
    if (
      ts.isClassDeclaration(p) ||
      ts.isClassExpression(p) ||
      ts.isInterfaceDeclaration(p) ||
      ts.isModuleDeclaration(p)
    ) {
      return p.name ? p.name.text : undefined;
    }
    p = p.parent;
  }
  return undefined;
}

function buildFqn(sourceFile, container, name) {
  const rel = relPath(sourceFile.fileName);
  return container ? `${rel}::${container}::${name}` : `${rel}::${name}`;
}

function getLine(sourceFile, node) {
  try {
    return sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile, false)).line + 1;
  } catch (_) {
    return 0;
  }
}

const stdout = process.stdout;

function emit(fqn, resolvedType, line) {
  if (!fqn || isSkippable(resolvedType)) return;
  stdout.write(JSON.stringify({ fqn, resolved_type: resolvedType, line }) + '\n');
}

// ── AST visitor ───────────────────────────────────────────────────────────────

function visitNode(sourceFile, node) {
  const line = getLine(sourceFile, node);

  if (ts.isFunctionDeclaration(node) && node.name && ts.isIdentifier(node.name)) {
    // Top-level function: emit return type.
    const sym = checker.getSymbolAtLocation(node.name);
    if (sym) {
      const type = checker.getTypeOfSymbolAtLocation(sym, node.name);
      const sigs = type.getCallSignatures();
      if (sigs.length > 0) {
        const ret = typeStr(checker.getReturnTypeOfSignature(sigs[0]));
        if (!isSkippable(ret)) {
          emit(buildFqn(sourceFile, undefined, node.name.text), ret, line);
        }
      }
    }

  } else if (
    (ts.isMethodDeclaration(node) || ts.isMethodSignature(node)) &&
    node.name && ts.isIdentifier(node.name)
  ) {
    // Class / interface method: emit return type under Container::method FQN.
    const sym = checker.getSymbolAtLocation(node.name);
    if (sym) {
      const type = checker.getTypeOfSymbolAtLocation(sym, node.name);
      const sigs = type.getCallSignatures();
      if (sigs.length > 0) {
        const ret = typeStr(checker.getReturnTypeOfSignature(sigs[0]));
        if (!isSkippable(ret)) {
          emit(buildFqn(sourceFile, containerName(node), node.name.text), ret, line);
        }
      }
    }

  } else if (ts.isVariableDeclaration(node) && node.name && ts.isIdentifier(node.name)) {
    // Variable / const: emit inferred type.
    const sym = checker.getSymbolAtLocation(node.name);
    if (sym) {
      const type = checker.getTypeOfSymbolAtLocation(sym, node.name);
      const t = typeStr(type);
      if (!isSkippable(t)) {
        emit(buildFqn(sourceFile, undefined, node.name.text), t, line);
      }
    }

  } else if (
    (ts.isPropertyDeclaration(node) || ts.isPropertySignature(node)) &&
    node.name && ts.isIdentifier(node.name)
  ) {
    // Class / interface property: emit type under Container::prop FQN.
    const sym = checker.getSymbolAtLocation(node.name);
    if (sym) {
      const type = checker.getTypeOfSymbolAtLocation(sym, node.name);
      const t = typeStr(type);
      if (!isSkippable(t)) {
        emit(buildFqn(sourceFile, containerName(node), node.name.text), t, line);
      }
    }

  } else if (ts.isClassDeclaration(node) && node.name) {
    // Class declaration: if it extends something, record the base types.
    const sym = checker.getSymbolAtLocation(node.name);
    if (sym) {
      const type = checker.getDeclaredTypeOfSymbol(sym);
      const bases = type.getBaseTypes ? type.getBaseTypes() : [];
      if (bases && bases.length > 0) {
        const impls = bases.map(typeStr).filter(s => !isSkippable(s)).join(', ');
        if (impls) {
          emit(buildFqn(sourceFile, undefined, node.name.text), 'extends ' + impls, line);
        }
      }
    }
  }

  ts.forEachChild(node, child => visitNode(sourceFile, child));
}

// ── Main ──────────────────────────────────────────────────────────────────────

const sourceFiles = program.getSourceFiles().filter(
  f => !f.isDeclarationFile &&
       !f.fileName.includes('node_modules') &&
       !f.fileName.includes('.codesurgeon')
);

for (const sf of sourceFiles) {
  ts.forEachChild(sf, node => visitNode(sf, node));
}
