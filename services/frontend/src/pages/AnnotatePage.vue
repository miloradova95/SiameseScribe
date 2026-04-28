<template>
  <div class="min-h-screen bg-[#f7f3ef] text-[#5b4033]">
    <main class="mx-auto w-full max-w-[1560px] px-6 py-6">
      <div v-if="image" class="grid grid-cols-[210px_minmax(0,1fr)] gap-8">
        <AnnotationSidebar
          :image="image"
          :image-src="mainImageSrc"
          :patches="patches"
          :selected-patch-id="selectedPatch?.id"
          :title="displayTitle"
          @back="router.push('/browse')"
          @select-patch="selectPatch"
        />

        <section class="rounded-[18px] border border-[#cfc5bc] bg-[#fbf8f5] p-8">
          <div class="grid grid-cols-[1fr_1fr_330px] gap-8">
            <div>
              <p class="mb-3 text-[13px] font-semibold">Selected Patch</p>

              <div class="overflow-hidden rounded-[10px] bg-[#ebe3db]">
                <img
                  :src="selectedPatchImage"
                  alt="Selected patch"
                  class="h-[275px] w-full object-cover"
                  @error="handleImageError"
                />
              </div>

              <p class="mt-3 text-center text-sm">
                {{ selectedPatchLabel }}
              </p>
            </div>

            <div>
              <p class="mb-3 text-[13px] font-semibold">Best Match</p>

              <div class="overflow-hidden rounded-[10px] bg-[#ebe3db]">
                <img
                  :src="bestMatch?.imageSrc || fallbackImage"
                  :alt="bestMatch?.label || 'Best match'"
                  class="h-[275px] w-full object-cover"
                  @error="handleImageError"
                />
              </div>

              <p class="mt-3 text-center text-sm">
                {{ bestMatch?.label || 'â€”' }}
              </p>
            </div>

            <div class="pt-8">
              <div class="mb-8">
                <div class="flex items-center gap-2 text-[14px] font-semibold">
                  <span>Similarity Score</span>
                  <span class="flex h-5 w-5 items-center justify-center rounded-full border border-[#6c4f3d] text-xs">
                    i
                  </span>
                </div>

                <p class="mt-1 text-[24px]">
                  {{ bestMatch?.score ?? 'â€”' }}%
                </p>
              </div>
            </div>
          </div>

          <div class="mt-8 border-t border-[#d8cec5] pt-5">
            <div class="flex items-center justify-between">
              <div>
                <div class="mb-3 flex items-center gap-2">
                  <p>How similar are these patches?</p>
                  <span class="flex h-5 w-5 items-center justify-center rounded-full border border-[#6c4f3d] text-xs">
                    i
                  </span>
                </div>

                <div class="flex gap-3">
                  <button
                    class="rounded-full border px-5 py-2 text-sm transition"
                    :class="feedbackButtonClass('similar')"
                    :disabled="feedbackSaving || !canSubmitFeedback"
                    @click="submitFeedback('similar')"
                  >
                    Similar
                  </button>
                  <button
                    class="rounded-full border px-5 py-2 text-sm transition"
                    :class="feedbackButtonClass('not_similar')"
                    :disabled="feedbackSaving || !canSubmitFeedback"
                    @click="submitFeedback('not_similar')"
                  >
                    Not Similar
                  </button>
                  <button
                    class="rounded-full border border-[#8a6755] px-5 py-2 text-sm"
                    :disabled="feedbackSaving"
                    @click="clearFeedbackState"
                  >
                    Uncertain
                  </button>
                </div>
              </div>

              <p
                class="text-sm"
                :class="feedbackError ? 'text-red-700' : 'text-[#7f6a5c]'"
              >
                {{ feedbackStatusText }}
              </p>
            </div>
          </div>

          <div class="mt-8 border-t border-[#d8cec5] pt-5">
            <div class="mb-5 flex items-center justify-between">
              <div class="flex items-center gap-2">
                <h2 class="text-[16px]">Most similar patches</h2>
                <span class="flex h-5 w-5 items-center justify-center rounded-full border border-[#6c4f3d] text-xs">
                  i
                </span>
              </div>

              <div class="flex items-center gap-3 text-[13px]">
                <span>Heatmap</span>
                <button
                  type="button"
                  class="relative h-6 w-11 rounded-full transition"
                  :class="heatmapOn ? 'bg-[#7b5a49]' : 'bg-[#d6ccc3]'"
                  @click="heatmapOn = !heatmapOn"
                >
                  <span
                    class="absolute top-1 h-4 w-4 rounded-full bg-white transition"
                    :class="heatmapOn ? 'left-6' : 'left-1'"
                  />
                </button>
              </div>
            </div>

            <div v-if="similarLoading" class="py-8 text-center text-sm text-[#7f6a5c]">
              Finding similar patches...
            </div>

            <div v-else-if="similarError" class="py-8 text-center text-sm text-red-700">
              {{ similarError }}
            </div>

            <div v-else-if="similarPatches.length" class="grid grid-cols-4 gap-8">
              <div
                v-for="patch in similarPatches"
                :key="patch.id"
                class="text-center"
              >
                <div class="overflow-hidden rounded-[8px] bg-[#ebe3db]">
                  <img
                    :src="patch.imageSrc"
                    :alt="patch.label"
                    class="h-[190px] w-full object-cover"
                    @error="handleImageError"
                  />
                </div>

                <p class="mt-2 text-[14px] font-semibold">
                  {{ patch.score }}%
                </p>

                <p class="text-[13px] text-[#7f6a5c]">
                  {{ patch.label }}
                </p>
              </div>
            </div>

            <div v-else class="py-8 text-center text-sm text-[#7f6a5c]">
              No similar patches found.
            </div>
          </div>
        </section>

        <div class="col-span-2 flex justify-end">
          <button class="rounded-full border border-[#8a6755] px-6 py-3 text-[15px]">
            Next Pen Flourish â€º
          </button>
        </div>
      </div>

      <div v-else-if="loading" class="py-20 text-center">
        Loading...
      </div>

      <div v-else class="py-20 text-center">
        Image not found.
      </div>
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AnnotationSidebar from '@/features/annotate/AnnotationSidebar.vue'
import { apiUrl } from '@/lib/api'
import { fetchImages } from '@/services/image-service'
import {
  fetchSimilarPatches,
  fetchPatchByFileName,
  fetchMyFeedbackForPair,
  getPatchFileUrlByName,
  fetchPatchesByImageId,
  patchName,
  saveFeedback,
} from '@/services/patch-service'

const route = useRoute()
const router = useRouter()

const image = ref(null)
const loading = ref(false)
const heatmapOn = ref(true)

const patches = ref([])
const selectedPatch = ref(null)

const similarPatches = ref([])
const similarLoading = ref(false)
const similarError = ref(null)
const feedbackSaving = ref(false)
const feedbackError = ref(null)
const lastFeedbackLabel = ref(null)
const feedbackSavedAt = ref(0)
const feedbackByPair = ref({})

const fileName = computed(() => route.params.fileName)

const fallbackImage =
  'data:image/svg+xml;charset=UTF-8,' +
  encodeURIComponent(`
    <svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">
      <rect width="100%" height="100%" fill="#ebe3db"/>
    </svg>
  `)

const displayTitle = computed(() => {
  if (!image.value?.fileName) return ''
  return image.value.fileName.replace(/\.[^.]+$/, '')
})

const selectedPatchLabel = computed(() => {
  if (selectedPatch.value?.patch_filename) return selectedPatch.value.patch_filename
  if (selectedPatch.value?.file_path) return patchName(selectedPatch.value.file_path)
  if (selectedPatch.value?.fileName) return selectedPatch.value.fileName
  if (selectedPatch.value?.label) return selectedPatch.value.label
  return displayTitle.value
})

const bestMatch = computed(() => similarPatches.value[0] || null)
const canSubmitFeedback = computed(() => Boolean(selectedPatch.value?.id && bestMatch.value?.id))
const feedbackStatusText = computed(() => {
  if (feedbackSaving.value) return 'Saving feedback...'
  if (feedbackError.value) return feedbackError.value
  if (lastFeedbackLabel.value && feedbackSavedAt.value) {
    return lastFeedbackLabel.value === 'similar'
      ? 'Saved as similar for this patch pair.'
      : 'Saved as not similar for this patch pair.'
  }
  if (!canSubmitFeedback.value) return 'Select a patch with a valid best match to save feedback.'
  return 'Click Similar or Not Similar to save feedback.'
})

const mainImageSrc = computed(() => {
  if (!image.value?.id) return fallbackImage
  return apiUrl(`/images/${image.value.id}/file`)
})

const selectedPatchImage = computed(() => {
  if (!selectedPatch.value) return fallbackImage

  if (selectedPatch.value.id) {
    return apiUrl(`/patches/${selectedPatch.value.id}/file`)
  }

  if (selectedPatch.value.patch_filename) {
    return getPatchFileUrlByName(selectedPatch.value.patch_filename)
  }

  if (selectedPatch.value.imageUrl) return selectedPatch.value.imageUrl
  if (selectedPatch.value.fileUrl) return selectedPatch.value.fileUrl

  return fallbackImage
})

async function selectPatch(patch) {
  selectedPatch.value = patch
  clearFeedbackState()
  await loadSimilarForPatch(patch)
}

function getPatchPath(patch) {
  return patch.file_path || patch.filePath || patch.patch_path || patch.patch_filename
}

function normalizePatch(patch) {
  return {
    ...patch,
    x: Number(patch.bbox?.x ?? patch.x ?? 0),
    y: Number(patch.bbox?.y ?? patch.y ?? 0),
    width: Number(patch.bbox?.width ?? patch.width ?? 128),
    height: Number(patch.bbox?.height ?? patch.height ?? 128),
  }
}

function handleImageError(event) {
  event.target.src = fallbackImage
}

function clearFeedbackState() {
  feedbackSaving.value = false
  feedbackError.value = null
  lastFeedbackLabel.value = null
  feedbackSavedAt.value = 0
}

function currentFeedbackKey() {
  if (!selectedPatch.value?.id || !bestMatch.value?.id) return null
  return `${selectedPatch.value.id}:${bestMatch.value.id}`
}

function applyFeedbackState(feedback) {
  lastFeedbackLabel.value = feedback?.label ?? null
  feedbackSavedAt.value = feedback ? Date.parse(feedback.created_at) || Date.now() : 0
}

function feedbackButtonClass(label) {
  const isActive = lastFeedbackLabel.value === label && feedbackSavedAt.value
  const isDisabled = feedbackSaving.value || !canSubmitFeedback.value

  if (isActive) {
    return 'border-[#6c4f3d] bg-[#6c4f3d] text-white'
  }

  if (isDisabled) {
    return 'border-[#c8baae] text-[#b49f91]'
  }

  return 'border-[#8a6755] text-[#5b4033] hover:bg-[#f0e7e0]'
}

async function submitFeedback(label) {
  if (!canSubmitFeedback.value || feedbackSaving.value) return

  feedbackSaving.value = true
  feedbackError.value = null

  try {
    await saveFeedback({
      queryPatchId: selectedPatch.value.id,
      resultPatchId: bestMatch.value.id,
      label,
    })
    const key = currentFeedbackKey()
    const savedFeedback = {
      query_patch_id: selectedPatch.value.id,
      result_patch_id: bestMatch.value.id,
      label,
      created_at: new Date().toISOString(),
    }

    if (key) {
      feedbackByPair.value = {
        ...feedbackByPair.value,
        [key]: savedFeedback,
      }
    }

    applyFeedbackState(savedFeedback)
  } catch (error) {
    feedbackError.value = error.message || 'Failed to save feedback.'
  } finally {
    feedbackSaving.value = false
  }
}

async function loadExistingFeedbackForCurrentPair() {
  const key = currentFeedbackKey()

  if (!key) {
    applyFeedbackState(null)
    return
  }

  if (Object.prototype.hasOwnProperty.call(feedbackByPair.value, key)) {
    applyFeedbackState(feedbackByPair.value[key])
    return
  }

  try {
    const feedback = await fetchMyFeedbackForPair({
      queryPatchId: selectedPatch.value.id,
      resultPatchId: bestMatch.value.id,
    })

    feedbackByPair.value = {
      ...feedbackByPair.value,
      [key]: feedback,
    }
    feedbackError.value = null
    applyFeedbackState(feedback)
  } catch (error) {
    feedbackError.value = error.message || 'Failed to load existing feedback.'
  }
}

async function loadSimilarForPatch(patch) {
  const path = getPatchPath(patch)

  if (!path) {
    similarPatches.value = []
    similarError.value = 'Selected patch has no file path.'
    return
  }

  similarLoading.value = true
  similarError.value = null
  similarPatches.value = []

  try {
    const results = await fetchSimilarPatches(path, { topK: 4 })
    const resolvedPatches = await Promise.all(
      results.map(async (item) => {
        const patchRecord = await fetchPatchByFileName(item.patch_filename)
        return {
          id: patchRecord.id,
          label: item.patch_filename,
          score: Math.round(item.similarity_score * 100),
          imageSrc: getPatchFileUrlByName(item.patch_filename),
        }
      })
    )

    similarPatches.value = resolvedPatches
    await loadExistingFeedbackForCurrentPair()
  } catch (error) {
    similarError.value = error.message || 'Failed to load similar patches.'
  } finally {
    similarLoading.value = false
  }
}

async function loadImageFromRoute() {
  loading.value = true
  image.value = null
  patches.value = []
  selectedPatch.value = null
  similarPatches.value = []
  similarError.value = null
  clearFeedbackState()

  try {
    const data = await fetchImages()
    const allImages = Array.isArray(data) ? data : []

    const found = allImages.find(
      (img) => img.fileName === decodeURIComponent(fileName.value)
    )

    image.value = found || null

    if (!image.value?.id) return

    const patchData = await fetchPatchesByImageId(image.value.id)

    patches.value = Array.isArray(patchData)
      ? patchData.map(normalizePatch)
      : []

    selectedPatch.value = patches.value[0] || null

    if (selectedPatch.value) {
      await loadSimilarForPatch(selectedPatch.value)
    }
  } catch (error) {
    console.error('Error loading image/patches:', error)
    similarError.value = error.message || 'Failed to load image.'
  } finally {
    loading.value = false
  }
}

onMounted(loadImageFromRoute)

watch(() => route.params.fileName, loadImageFromRoute)
</script>
