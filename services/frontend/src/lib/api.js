import { useAuthStore } from '../stores/auth'

const API_BASE = import.meta.env.DEV
  ? '/api'
  : (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000')

export function apiUrl(path) {
  return `${API_BASE}${path}`
}

export async function fetchWithAuth(url, options = {}) {
  const authStore = useAuthStore()
  const headers = {
    ...options.headers,
    ...(authStore.token ? { Authorization: `Bearer ${authStore.token}` } : {}),
  }
  const response = await fetch(url, { ...options, headers })
  if (response.status === 401) {
    authStore.logout()
  }
  return response
}
