import { apiUrl, fetchWithAuth } from '../lib/api'
import { useAuthStore } from '../stores/auth'

let bookmarkCache = null
let bookmarkRequest = null

function buildErrorMessage(prefix, response) {
  return `${prefix} (${response.status})`
}

function emitBookmarksUpdated(bookmarks) {
  window.dispatchEvent(
    new CustomEvent('bookmarks-updated', {
      detail: { bookmarks },
    })
  )
}

function getCurrentUserId() {
  const authStore = useAuthStore()
  return authStore.user?.id ?? null
}

export function clearBookmarkCache() {
  bookmarkCache = null
  bookmarkRequest = null
}

export async function fetchBookmarks(force = false) {
  const userId = getCurrentUserId()
  if (!userId) {
    bookmarkCache = []
    return bookmarkCache
  }

  if (!force && bookmarkCache) {
    return bookmarkCache
  }

  if (!force && bookmarkRequest) {
    return bookmarkRequest
  }

  bookmarkRequest = fetchWithAuth(apiUrl(`/users/${userId}/bookmarks`))
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(buildErrorMessage('Failed to fetch bookmarks', response))
      }

      const bookmarks = await response.json()
      bookmarkCache = Array.isArray(bookmarks) ? bookmarks : []
      return bookmarkCache
    })
    .finally(() => {
      bookmarkRequest = null
    })

  return bookmarkRequest
}

export async function addBookmark(imageId) {
  const userId = getCurrentUserId()
  if (!userId) {
    throw new Error('No active user for bookmark request')
  }

  const response = await fetchWithAuth(apiUrl(`/users/${userId}/bookmarks/${imageId}`), {
    method: 'POST',
  })

  if (!response.ok) {
    throw new Error(buildErrorMessage('Failed to save bookmark', response))
  }

  const bookmarks = await response.json()
  bookmarkCache = Array.isArray(bookmarks) ? bookmarks : []
  emitBookmarksUpdated(bookmarkCache)
  return bookmarkCache
}

export async function removeBookmark(imageId) {
  const userId = getCurrentUserId()
  if (!userId) {
    throw new Error('No active user for bookmark request')
  }

  const response = await fetchWithAuth(apiUrl(`/users/${userId}/bookmarks/${imageId}`), {
    method: 'DELETE',
  })

  if (!response.ok) {
    throw new Error(buildErrorMessage('Failed to remove bookmark', response))
  }

  const bookmarks = await response.json()
  bookmarkCache = Array.isArray(bookmarks) ? bookmarks : []
  emitBookmarksUpdated(bookmarkCache)
  return bookmarkCache
}
