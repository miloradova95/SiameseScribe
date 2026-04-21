const API_BASE = import.meta.env.DEV
  ? '/api'
  : (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000')

export function apiUrl(path) {
  return `${API_BASE}${path}`
}
