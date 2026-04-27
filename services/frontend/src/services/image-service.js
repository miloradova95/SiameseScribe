import { apiUrl } from '../lib/api'
import { fetchPatchesByImageId } from './patch-service'

function buildErrorMessage(prefix, response) {
  return `${prefix} (${response.status})`
}

export function getImageFileUrl(imageId) {
  return apiUrl(`/images/${imageId}/file`)
}

export async function fetchImages() {
  const response = await fetch(apiUrl('/images'))
  if (!response.ok) {
    throw new Error(buildErrorMessage('Failed to fetch images', response))
  }

  return response.json()
}

export async function fetchRandomImage() {
  const response = await fetch(apiUrl('/images/random'))
  if (!response.ok) {
    throw new Error(buildErrorMessage('Failed to fetch random image', response))
  }

  return response.json()
}

export async function uploadImage(file, group) {
  const formData = new FormData()
  formData.append('file', file)

  if (group?.trim()) {
    formData.append('group', group.trim())
  }

  const response = await fetch(apiUrl('/images/upload'), {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(`Upload failed (${response.status}): ${detail}`)
  }

  return response.json()
}

export async function runUploadPipeline(file, group) {
  const image = await uploadImage(file, group)
  const patches = image?.id ? await fetchPatchesByImageId(image.id) : []

  return {
    image,
    patches,
  }
}
