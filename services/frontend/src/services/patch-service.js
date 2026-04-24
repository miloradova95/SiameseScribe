import { apiUrl } from '../lib/api'

const ML_API = import.meta.env.VITE_ML_API_URL ?? 'http://localhost:8001'

function buildErrorMessage(prefix, response) {
  return `${prefix} (${response.status})`
}

export function patchName(filePath) {
  return filePath?.split(/[\\/]/).pop() ?? filePath
}

export function getPatchFileUrl(patchId) {
  return apiUrl(`/patches/${patchId}/file`)
}

export function getPatchFileUrlByName(fileName) {
  return apiUrl(`/patches/by-file-name/${encodeURIComponent(fileName)}/file`)
}

export async function fetchPatchByFileName(fileName) {
  const response = await fetch(apiUrl(`/patches/by-file-name/${encodeURIComponent(fileName)}`))
  if (!response.ok) {
    throw new Error(buildErrorMessage('Patch not found', response))
  }

  return response.json()
}

export async function fetchPatchesByImageId(imageId) {
  const response = await fetch(apiUrl(`/images/${imageId}/patches`))
  if (!response.ok) {
    throw new Error(buildErrorMessage('Failed to fetch patches', response))
  }

  return response.json()
}

export async function fetchSimilarPatches(filePath, options = {}) {
  const { topK = 8 } = options

  const embedResponse = await fetch(`${ML_API}/embed_patches`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ patch_paths: [filePath] }),
  })
  if (!embedResponse.ok) {
    throw new Error(buildErrorMessage('Embedding failed', embedResponse))
  }

  const embedData = await embedResponse.json()
  const vector = embedData.embeddings?.[0]?.vector
  if (!vector) {
    throw new Error('No embedding returned')
  }

  const searchResponse = await fetch(`${ML_API}/search_patches`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ embedding: vector, top_k: topK }),
  })
  if (!searchResponse.ok) {
    throw new Error(buildErrorMessage('Search failed', searchResponse))
  }

  const searchData = await searchResponse.json()
  const queryFileName = patchName(filePath)

  return (searchData.results ?? []).filter((result) => result.patch_filename !== queryFileName)
}
