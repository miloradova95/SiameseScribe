import { apiUrl, fetchWithAuth } from '../lib/api'

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
  const response = await fetchWithAuth(apiUrl(`/patches/by-file-name/${encodeURIComponent(fileName)}`))
  if (!response.ok) {
    throw new Error(buildErrorMessage('Patch not found', response))
  }

  return response.json()
}

export async function fetchPatchesByImageId(imageId) {
  const response = await fetchWithAuth(apiUrl(`/images/${imageId}/patches`))
  if (!response.ok) {
    throw new Error(buildErrorMessage('Failed to fetch patches', response))
  }

  return response.json()
}

export async function fetchSimilarPatches(filePath, options = {}) {
  const { topK = 8, sourceImageId = null } = options

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

  const searchBody = { embedding: vector, top_k: topK }
  if (sourceImageId != null) {
    searchBody.exclude_source_image_id = sourceImageId
  }

  const searchResponse = await fetch(`${ML_API}/search_patches`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(searchBody),
  })
  if (!searchResponse.ok) {
    throw new Error(buildErrorMessage('Search failed', searchResponse))
  }

  const searchData = await searchResponse.json()
  const queryFileName = patchName(filePath)

  return (searchData.results ?? []).filter((result) => result.patch_filename !== queryFileName)
}

export async function fetchExplainPair(queryPatchPath, resultPatchPath) {
  const response = await fetch(`${ML_API}/explain_pair`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query_patch_path: queryPatchPath, result_patch_path: resultPatchPath }),
  })
  if (!response.ok) {
    throw new Error(buildErrorMessage('Heatmap generation failed', response))
  }
  return response.json()
}

export function getHeatmapFileUrl(relativePath) {
  const filename = relativePath.split('/').pop()
  return apiUrl(`/heatmaps/${encodeURIComponent(filename)}`)
}

export async function saveFeedback({ queryPatchId, resultPatchId, label }) {
  const response = await fetchWithAuth(apiUrl('/feedback'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query_patch_id: queryPatchId,
      result_patch_id: resultPatchId,
      label,
    }),
  })

  if (!response.ok) {
    throw new Error(buildErrorMessage('Failed to save feedback', response))
  }

  return response.json()
}

export async function fetchMyFeedback() {
  const response = await fetchWithAuth(apiUrl('/feedback/mine'))
  if (!response.ok) {
    throw new Error(buildErrorMessage('Failed to fetch feedback', response))
  }

  return response.json()
}

export async function fetchMyFeedbackForPair({ queryPatchId, resultPatchId }) {
  const response = await fetchWithAuth(
    apiUrl(
      `/feedback/mine/by-pair?query_patch_id=${encodeURIComponent(queryPatchId)}&result_patch_id=${encodeURIComponent(resultPatchId)}`
    )
  )
  if (!response.ok) {
    throw new Error(buildErrorMessage('Failed to fetch feedback', response))
  }

  return response.json()
}

export async function fetchAdminFeedback(filters = {}) {
  const params = new URLSearchParams()

  if (filters.userId) params.set('user_id', String(filters.userId))
  if (filters.dateFrom) params.set('date_from', filters.dateFrom)
  if (filters.dateTo) params.set('date_to', filters.dateTo)
  if (filters.usedForRetrain !== '' && filters.usedForRetrain !== null && filters.usedForRetrain !== undefined) {
    params.set('used_for_retrain', String(filters.usedForRetrain))
  }

  const query = params.toString()
  const response = await fetchWithAuth(apiUrl(`/feedback/admin${query ? `?${query}` : ''}`))
  if (!response.ok) {
    throw new Error(buildErrorMessage('Failed to fetch admin feedback', response))
  }

  return response.json()
}

// retrainFromFeedback() was removed. Finetuning is now triggered automatically
// by the backend scheduler, or manually via triggerFinetuneRun() below.

export async function triggerFinetuneRun() {
  const response = await fetchWithAuth(apiUrl('/admin/finetune-runs/trigger'), {
    method: 'POST',
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({}))
    throw new Error(err?.detail ?? `Trigger failed (${response.status})`)
  }
  return response.json()
}

export async function fetchFinetuneRuns(limit = 50) {
  const response = await fetchWithAuth(apiUrl(`/admin/finetune-runs?limit=${limit}`))
  if (!response.ok) {
    throw new Error(`Failed to load finetune runs (${response.status})`)
  }
  return response.json()
}

export async function fetchReembedStatus() {
  const ML_API = import.meta.env.VITE_ML_API_URL ?? 'http://localhost:8001'
  const response = await fetch(`${ML_API}/reembed_status`)
  if (!response.ok) {
    throw new Error(`Failed to fetch re-embed status (${response.status})`)
  }
  return response.json()
}


