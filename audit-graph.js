/**
 * audit-graph.js
 * Tool Audit Dependency Pattern: Traces dependency chains from any target file
 * upward to entry points (src/index.js, src/app/bootstrap.js) and audits
 * parallel branches, layer depth, and acyclic DAG integrity.
 */

import fs from 'node:fs';
import path from 'node:path';

const SRC_DIR = path.resolve('src');
const ENTRY_POINTS = [
  'src/index.js',
  'src/app/bootstrap.js',
];

// 1. Collect all JS files in src/
function getAllFiles(dir) {
  let files = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files = files.concat(getAllFiles(fullPath));
    } else if (entry.isFile() && entry.name.endsWith('.js')) {
      files.push(path.relative(process.cwd(), fullPath));
    }
  }
  return files;
}

// 2. Parse imports from file content
function extractDependencies(filePath) {
  const code = fs.readFileSync(filePath, 'utf-8');
  const dir = path.dirname(filePath);
  const imports = new Set();

  // Match: import ... from './path.js' or export ... from './path.js'
  const importExportRegex = /(?:import|export)\s+(?:[\s\S]*?from\s+)?['"](\.[^'"]+)['"]/g;
  let match;
  while ((match = importExportRegex.exec(code)) !== null) {
    const relSpecifier = match[1];
    const resolved = path.normalize(path.join(dir, relSpecifier));
    imports.add(resolved);
  }
  return Array.from(imports);
}

// 3. Build Full Dependency Graph (Forward: file -> dependencies, Reverse: file -> dependents)
export function buildGraph() {
  const allFiles = getAllFiles(SRC_DIR);
  const forwardGraph = new Map(); // file -> [dependencies it imports]
  const reverseGraph = new Map(); // file -> [dependents that import it]

  for (const f of allFiles) {
    forwardGraph.set(f, []);
    reverseGraph.set(f, []);
  }

  for (const f of allFiles) {
    const deps = extractDependencies(f);
    forwardGraph.set(f, deps);
    for (const d of deps) {
      if (!reverseGraph.has(d)) reverseGraph.set(d, []);
      reverseGraph.get(d).push(f);
    }
  }

  return { allFiles, forwardGraph, reverseGraph };
}

// 4. Trace upward paths from target file to entry points
export function traceUpwardToEntryPoints(targetFile, reverseGraph, entryPoints = ENTRY_POINTS) {
  const paths = [];

  function dfs(current, currentPath, visited) {
    if (entryPoints.includes(current)) {
      paths.push([...currentPath]);
      return;
    }
    const parents = reverseGraph.get(current) || [];
    for (const parent of parents) {
      if (!visited.has(parent)) {
        visited.add(parent);
        currentPath.push(parent);
        dfs(parent, currentPath, visited);
        currentPath.pop();
        visited.delete(parent);
      }
    }
  }

  dfs(targetFile, [targetFile], new Set([targetFile]));
  return paths;
}

// 5. Detect cycles (Kahn's or DFS)
export function detectCycles(allFiles, forwardGraph) {
  const visited = new Map(); // 0 = unvisited, 1 = visiting, 2 = visited
  const cycle = [];

  function dfs(node, stack) {
    visited.set(node, 1);
    stack.push(node);
    for (const dep of forwardGraph.get(node) || []) {
      if (visited.get(dep) === 1) {
        cycle.push([...stack, dep]);
      } else if (!visited.has(dep) || visited.get(dep) === 0) {
        dfs(dep, stack);
      }
    }
    stack.pop();
    visited.set(node, 2);
  }

  for (const f of allFiles) {
    if (!visited.has(f) || visited.get(f) === 0) {
      dfs(f, []);
    }
  }
  return cycle;
}

// 6. Calculate Layer / Topological Height
export function calculateLayers(allFiles, forwardGraph) {
  const memo = new Map();

  function getHeight(node, visited = new Set()) {
    if (memo.has(node)) return memo.get(node);
    if (visited.has(node)) return 0; // avoid infinite loop if cyclic
    visited.add(node);

    const deps = forwardGraph.get(node) || [];
    if (deps.length === 0) {
      memo.set(node, 0);
      return 0;
    }

    let maxHeight = 0;
    for (const dep of deps) {
      maxHeight = Math.max(maxHeight, getHeight(dep, new Set(visited)) + 1);
    }
    memo.set(node, maxHeight);
    return maxHeight;
  }

  const layers = new Map();
  for (const f of allFiles) {
    layers.set(f, getHeight(f));
  }
  return layers;
}

// CLI Execution
export function runCliAudit(targetFilter) {
  const { allFiles, forwardGraph, reverseGraph } = buildGraph();
  const cycles = detectCycles(allFiles, forwardGraph);
  const layers = calculateLayers(allFiles, forwardGraph);

  console.log('\n===============================================================');
  console.log('       AUDIT ALUR PARALEL DEPENDENCY PATTERN HINGGA ENTRY      ');
  console.log('===============================================================');
  console.log(`Total Module Terdeteksi: ${allFiles.length} file`);
  console.log(`Status Siklus (Circular Dependency): ${cycles.length === 0 ? '\x1b[32m0 SIKLUS (Clean DAG)\x1b[0m' : `\x1b[31m${cycles.length} SIKLUS TERDETEKSI\x1b[0m`}`);
  console.log(`Primary Entry Points: ${ENTRY_POINTS.join(', ')}\n`);

  if (targetFilter) {
    const matched = allFiles.filter(f => f.includes(targetFilter));
    if (matched.length === 0) {
      console.log(`\x1b[31mFile target "${targetFilter}" tidak ditemukan!\x1b[0m`);
      return;
    }

    for (const target of matched) {
      console.log(`\x1b[1m\x1b[36m>>> AUDIT TARGET FILE: ${target}\x1b[0m`);
      console.log(`    - Level / Layer Depth: Layer ${layers.get(target)} (0 = Dasar/Leaf)`);
      
      const directDeps = forwardGraph.get(target) || [];
      console.log(`    - Direct Dependencies (${directDeps.length}):`);
      if (directDeps.length === 0) {
        console.log(`      * (None - Independent Leaf Module)`);
      } else {
        directDeps.forEach((d, idx) => {
          console.log(`      ${idx + 1}. ${d} [Layer ${layers.get(d)}]`);
        });
      }

      // Sibling / Parallel Execution Branches:
      // Dependents that import this target
      const dependents = reverseGraph.get(target) || [];
      console.log(`    - Direct Dependents / Importers (${dependents.length}):`);
      if (dependents.length === 0) {
        console.log(`      * (None - Top-level Root or Unused)`);
      } else {
        dependents.forEach((dep, idx) => {
          const peers = (forwardGraph.get(dep) || []).filter(p => p !== target);
          console.log(`      ${idx + 1}. \x1b[33m${dep}\x1b[0m`);
          if (peers.length > 0) {
            console.log(`         \x1b[90m↳ Paralel di-import bersamaan dengan: ${peers.join(', ')}\x1b[0m`);
          }
        });
      }

      // Upward Paths to Entry Points
      console.log(`    - Alur Rantai ke Entry Point:`);
      const paths = traceUpwardToEntryPoints(target, reverseGraph);
      if (paths.length === 0) {
        console.log(`      \x1b[33m* Tidak ada alur langsung ke entry point (Module terisolasi/standalone)\x1b[0m`);
      } else {
        paths.forEach((p, idx) => {
          console.log(`      \x1b[32mPath ${idx + 1}:\x1b[0m ${p.join(' \x1b[32m➔\x1b[0m ')}`);
        });
      }
      console.log('---------------------------------------------------------------');
    }
  } else {
    // Group files by architectural layer
    const maxLayer = Math.max(...Array.from(layers.values()));
    for (let l = 0; l <= maxLayer; l++) {
      const layerFiles = allFiles.filter(f => layers.get(f) === l);
      console.log(`\x1b[1m[LAYER ${l}] (${layerFiles.length} file paralel): \x1b[0m`);
      for (const f of layerFiles) {
        const upPaths = traceUpwardToEntryPoints(f, reverseGraph);
        const entryStatus = upPaths.length > 0 
          ? `\x1b[32m✓ Terhubung ke entry (${upPaths.length} path)\x1b[0m` 
          : (ENTRY_POINTS.includes(f) ? '\x1b[36m★ Entry Point\x1b[0m' : '\x1b[90mStandalone\x1b[0m');
        console.log(`  • ${f.padEnd(32)} ${entryStatus}`);
      }
      console.log('');
    }

    console.log('Untuk audit detail per satu file target spesifik:');
    console.log('  \x1b[33mnode audit-graph.js --file <nama_file>\x1b[0m (contoh: node audit-graph.js --file dtype.js)\n');
  }
}

// If invoked directly from CLI
const args = process.argv.slice(2);
const fileArgIndex = args.indexOf('--file');
const filter = fileArgIndex !== -1 && args[fileArgIndex + 1] ? args[fileArgIndex + 1] : null;
runCliAudit(filter);
