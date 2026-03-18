/* eslint-disable no-console */
/*
  Workaround for Windows environments where spawning `net use` is blocked (EPERM).

  Vite calls `exec("net use", ...)` inside its win32 safeRealPath optimization.
  When blocked, Vite fails to start with:
    [plugin externalize-deps] Error: spawn EPERM

  This patch removes that `net use` call and falls back to `fs.realpathSync.native`.
  It's safe for local dev; it only affects how Vite maps UNC shares to drive letters.
*/

const fs = require('fs')
const path = require('path')

function patchFile(filePath) {
  const src = fs.readFileSync(filePath, 'utf8')
  if (src.includes('PATCHED_NO_NET_USE')) return false

  const needle = 'exec("net use", (error, stdout) => {'
  const idx = src.indexOf(needle)
  if (idx === -1) return false

  // Insert a short-circuit right before the `net use` exec call.
  const injected =
    '/* PATCHED_NO_NET_USE: avoid spawn EPERM on locked-down Windows */\n' +
    '\tsafeRealpathSync = fs.realpathSync.native;\n' +
    '\treturn;\n' +
    '\t' +
    needle

  const next = src.slice(0, idx) + injected + src.slice(idx + needle.length)
  fs.writeFileSync(filePath, next, 'utf8')
  return true
}

function main() {
  const viteNodeChunk = path.join(
    process.cwd(),
    'node_modules',
    'vite',
    'dist',
    'node',
    'chunks',
    'node.js'
  )

  if (!fs.existsSync(viteNodeChunk)) {
    return
  }

  const changed = patchFile(viteNodeChunk)
  if (changed) {
    console.log('[postinstall] Patched Vite win32 safeRealPath (disabled `net use`).')
  }
}

main()

