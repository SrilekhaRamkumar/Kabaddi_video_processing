import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

function mountCrashOverlay(err) {
  const root = document.getElementById('root')
  if (!root) return
  const message =
    err instanceof Error
      ? `${err.name}: ${err.message}\n\n${err.stack || ''}`
      : String(err)

  root.innerHTML = `
    <div style="min-height:100vh;padding:24px;font-family:ui-monospace,Menlo,Consolas,monospace;background:#0b1220;color:#e2e8f0;">
      <div style="max-width:1000px;margin:0 auto;">
        <div style="font-size:14px;opacity:.85;margin-bottom:10px;">Kabaddi Frontend crashed</div>
        <pre style="white-space:pre-wrap;line-height:1.35;background:rgba(255,255,255,0.06);padding:16px;border-radius:12px;border:1px solid rgba(148,163,184,0.2);">${message.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')}</pre>
        <div style="margin-top:12px;font-size:12px;opacity:.8;">Open DevTools Console for more details.</div>
      </div>
    </div>
  `
}

window.addEventListener('error', (e) => {
  // Some errors don't populate `error`, fall back to message.
  mountCrashOverlay(e.error || e.message || 'Unknown error')
})

window.addEventListener('unhandledrejection', (e) => {
  mountCrashOverlay(e.reason || 'Unhandled promise rejection')
})

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
