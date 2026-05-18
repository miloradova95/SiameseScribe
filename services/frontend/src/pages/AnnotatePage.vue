<template>
  <div class="min-h-screen bg-[#f7f3ef] text-[#6c4f3d]">
    <main class="mx-auto max-w-[1560px] px-8 pb-10 pt-4">
      <section class="mt-2 border-b border-[#ddd3ca] pb-6">
        <div class="flex items-center gap-3 text-[18px]">
          <button
            class="text-[#c3b7ad] transition hover:text-[#8b6b59]"
            @click="router.push('/browse')"
          >
            Browse
          </button>
          <span class="text-[#d0c5bc]">/</span>
          <span class="font-medium text-[#6c4f3d]">{{ displayTitle || 'Annotate' }}</span>
        </div>
        <h1 class="mt-4 text-[28px] font-semibold text-[#6c4f3d]">Most similar patches</h1>
      </section>
      <div v-if="image" class="mt-6 grid grid-cols-[210px_minmax(0,1fr)] gap-8">
        <div>
          <AnnotationSidebar
            :image="image"
            :image-src="mainImageSrc"
            :patches="patches"
            :annotated-patch-ids="annotatedPatchIds"
            :annotated-patch-names="annotatedPatchNames"
            :selected-patch-id="selectedPatch?.id"
            :title="displayTitle"
            @back="router.push('/browse')"
            @select-patch="selectPatch"
          />

          <div class="mt-4 rounded-[16px] border border-[#cfc5bc] bg-[#fbf8f5] px-3 py-3">
            <div class="mb-2 flex items-baseline justify-between gap-2">
              <p class="text-[14px] font-semibold">Navigate Patches</p>
              <p class="text-[14px] text-[#7f6a5c]">
                {{ patches.length ? `${selectedPatchIndex + 1} / ${patches.length}` : '-' }}
              </p>
            </div>

            <div class="mb-2 flex gap-2">
              <button
                class="flex-1 rounded-full border border-[#8a6755] bg-[#8a6755] px-2.5 py-1 text-[13px] font-bold text-white transition hover:bg-[#6c4f3d] hover:border-[#6c4f3d] disabled:border-[#c8baae] disabled:bg-[#c8baae]"
                :disabled="!patches.length"
                @click="goToPrevPatch"
              >
                ← Prev
              </button>
              <button
                class="flex-1 rounded-full border border-[#8a6755] bg-[#8a6755] px-2.5 py-1 text-[13px] font-bold text-white transition hover:bg-[#6c4f3d] hover:border-[#6c4f3d] disabled:border-[#c8baae] disabled:bg-[#c8baae]"
                :disabled="!patches.length"
                @click="goToNextPatch"
              >
                Next →
              </button>
            </div>

            <div class="flex gap-2">
              <input
                v-model="jumpToId"
                type="number"
                min="1"
                :max="patches.length"
                placeholder="Patch #"
                class="w-0 flex-1 rounded-full border border-[#c8baae] bg-white px-3 py-1 text-[13px] text-[#5b4033] placeholder-[#b49f91] outline-none focus:border-[#8a6755]"
                @keydown.enter="jumpToPatch"
              />
              <button
                class="rounded-full border border-[#8a6755] bg-[#8a6755] px-3 py-1 text-[13px] font-bold text-white transition hover:bg-[#6c4f3d] hover:border-[#6c4f3d] disabled:border-[#c8baae] disabled:bg-[#c8baae]"
                :disabled="!patches.length"
                @click="jumpToPatch"
              >
                Go
              </button>
            </div>
          </div>
        </div>

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

              <button
                type="button"
                class="relative block w-full overflow-hidden rounded-[10px] border-2 border-transparent bg-[#ebe3db] transition hover:border-[#8a6755] focus:outline-none focus:ring-2 focus:ring-[#8a6755]"
                :disabled="!bestMatch"
                @click="openPatch(bestMatch)"
              >
                <AnnotatedBadge
                  v-if="patchHasFeedbackWithSelected(bestMatch)"
                  size="md"
                />
                <img
                  :src="
                    heatmapOnBestMatch && bestMatch?.heatmapSrc
                      ? bestMatch.heatmapSrc
                      : bestMatch?.imageSrc || fallbackImage
                  "
                  :alt="bestMatch?.label || 'Best match'"
                  class="h-[275px] w-full object-cover"
                  @error="handleImageError"
                />
              </button>

              <p class="mt-3 text-center text-sm">
                {{ bestMatch?.label || '-' }}
              </p>
            </div>

            <div class="pt-8">
              <div class="mb-6">
                <div class="flex items-center justify-between gap-2">
                  <div class="flex items-center gap-2 text-[14px] font-semibold">
                    <span>Similarity Score</span>
                    <span
                      class="flex h-5 w-5 items-center justify-center rounded-full border border-[#6c4f3d] text-xs"
                    >
                      i
                    </span>
                  </div>

                  <div class="flex items-center gap-2 text-[13px]">
                    <span>Heatmap</span>
                    <button
                      type="button"
                      class="relative h-6 w-11 rounded-full transition"
                      :class="heatmapOnBestMatch ? 'bg-[#7b5a49]' : 'bg-[#d6ccc3]'"
                      @click="heatmapOnBestMatch = !heatmapOnBestMatch"
                    >
                      <span
                        class="absolute top-1 h-4 w-4 rounded-full bg-white transition"
                        :class="heatmapOnBestMatch ? 'left-6' : 'left-1'"
                      />
                    </button>
                  </div>
                </div>

                <p class="mt-1 text-[24px]">{{ bestMatch?.score ?? '-' }}%</p>
              </div>

              <div v-if="false" class="hidden border-t border-[#d8cec5] pt-5">
                <p class="mb-2 text-[13px] font-semibold">Navigate Patches</p>
                <p class="mb-3 text-[12px] text-[#7f6a5c]">
                  {{ patches.length ? `${selectedPatchIndex + 1} / ${patches.length}` : '—' }}
                </p>

                <div class="mb-3 flex gap-2">
                  <button
                    class="flex-1 rounded-full border border-[#8a6755] bg-[#8a6755] px-3 py-1.5 text-sm font-bold text-white transition hover:bg-[#6c4f3d] hover:border-[#6c4f3d] disabled:border-[#c8baae] disabled:bg-[#c8baae]"
                    :disabled="!patches.length"
                    @click="goToPrevPatch"
                  >
                    ← Prev
                  </button>
                  <button
                    class="flex-1 rounded-full border border-[#8a6755] bg-[#8a6755] px-3 py-1.5 text-sm font-bold text-white transition hover:bg-[#6c4f3d] hover:border-[#6c4f3d] disabled:border-[#c8baae] disabled:bg-[#c8baae]"
                    :disabled="!patches.length"
                    @click="goToNextPatch"
                  >
                    Next →
                  </button>
                </div>

                <div class="flex gap-2">
                  <input
                    v-model="jumpToId"
                    type="number"
                    min="1"
                    :max="patches.length"
                    placeholder="Patch #"
                    class="w-0 flex-1 rounded-full border border-[#c8baae] bg-white px-3 py-1.5 text-sm text-[#5b4033] placeholder-[#b49f91] outline-none focus:border-[#8a6755]"
                    @keydown.enter="jumpToPatch"
                  />
                  <button
                    class="rounded-full border border-[#8a6755] bg-[#8a6755] px-4 py-1.5 text-sm font-bold text-white transition hover:bg-[#6c4f3d] hover:border-[#6c4f3d] disabled:border-[#c8baae] disabled:bg-[#c8baae]"
                    :disabled="!patches.length"
                    @click="jumpToPatch"
                  >
                    Go
                  </button>
                </div>
              </div>
            </div>
          </div>

          <div class="mt-8 border-t border-[#d8cec5] pt-5">
            <div class="flex items-center justify-between">
              <div>
                <div class="mb-3 flex items-center gap-2">
                  <p>How similar are these patches?</p>
                  <span
                    class="flex h-5 w-5 items-center justify-center rounded-full border border-[#6c4f3d] text-xs"
                  >
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

              <p class="text-sm" :class="feedbackError ? 'text-red-700' : 'text-[#7f6a5c]'">
                {{ feedbackStatusText }}
              </p>
            </div>
          </div>

          <div class="mt-8 border-t border-[#d8cec5] pt-5">
            <div class="mb-5 flex items-center justify-between">
              <div class="flex items-center gap-2">
                <h2 class="text-[16px]">Other similar patches</h2>
                <span
                  class="flex h-5 w-5 items-center justify-center rounded-full border border-[#6c4f3d] text-xs"
                >
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

            <div v-else-if="otherSimilarPatches.length" class="grid grid-cols-4 gap-8">
              <div v-for="patch in otherSimilarPatches" :key="patch.id" class="text-center">
                <button
                  type="button"
                  class="relative block w-full overflow-hidden rounded-[8px] border-2 border-transparent bg-[#ebe3db] transition hover:border-[#8a6755] focus:outline-none focus:ring-2 focus:ring-[#8a6755]"
                  @click="selectBestMatch(patch)"
                >
                  <AnnotatedBadge v-if="patchHasFeedbackWithSelected(patch)" />
                  <img
                    :src="heatmapOn && patch.heatmapSrc ? patch.heatmapSrc : patch.imageSrc"
                    :alt="patch.label"
                    class="h-[190px] w-full object-cover"
                    @error="handleImageError"
                  />
                </button>

                <p class="mt-2 text-[14px] font-semibold">{{ patch.score }}%</p>

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
      </div>

      <div v-else-if="loading" class="py-20 text-center">Loading...</div>

      <div v-else class="py-20 text-center">Image not found.</div>
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import AnnotatedBadge from '@/features/annotate/AnnotatedBadge.vue'
import AnnotationSidebar from '@/features/annotate/AnnotationSidebar.vue'
import { apiUrl } from '@/lib/api'
import { fetchImages } from '@/services/image-service'
import {
  fetchSimilarPatches,
  fetchPatchByFileName,
  fetchMyFeedbackForPair,
  fetchMyFeedback,
  getPatchFileUrlByName,
  fetchPatchesByImageId,
  patchName,
  saveFeedback,
  fetchExplainPair,
  getHeatmapFileUrl,
} from '@/services/patch-service'

const route = useRoute()
const router = useRouter()

const image = ref(null)
const loading = ref(false)
const heatmapOn = ref(false) // toggles heatmaps for "Other similar patches" grid
const heatmapOnBestMatch = ref(false) // toggles heatmap for "Best Match" only

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
const annotatedPatchIds = ref([])
const annotatedPatchNames = ref([])
const jumpToId = ref('')

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
const otherSimilarPatches = computed(() => similarPatches.value.slice(1))

const selectedPatchIndex = computed(() => {
  if (!selectedPatch.value || !patches.value.length) return -1
  return patches.value.findIndex((p) => p.id === selectedPatch.value.id)
})
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

function goToNextPatch() {
  if (!patches.value.length) return
  const idx = selectedPatchIndex.value
  const next = idx < patches.value.length - 1 ? idx + 1 : 0
  selectPatch(patches.value[next])
}

function goToPrevPatch() {
  if (!patches.value.length) return
  const idx = selectedPatchIndex.value
  const prev = idx > 0 ? idx - 1 : patches.value.length - 1
  selectPatch(patches.value[prev])
}

function jumpToPatch() {
  const num = parseInt(jumpToId.value, 10)
  if (isNaN(num) || num < 1 || num > patches.value.length) return
  selectPatch(patches.value[num - 1])
  jumpToId.value = ''
}

function patchRouteParam(patch) {
  return patch?.label || patch?.patch_filename || patchName(getPatchPath(patch))
}

function openPatch(patch) {
  const target = patchRouteParam(patch)
  if (!target) return
  router.push(`/browse/${encodeURIComponent(target)}`)
}

async function selectBestMatch(patch) {
  const selectedIndex = similarPatches.value.findIndex((item) => item.id === patch.id)
  if (selectedIndex <= 0) return

  const updatedPatches = [...similarPatches.value]
  updatedPatches[selectedIndex] = updatedPatches[0]
  updatedPatches[0] = patch
  similarPatches.value = updatedPatches

  clearFeedbackState()
  await loadExistingFeedbackForCurrentPair()
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

function showFeedbackToast(message, type = 'success') {
  window.dispatchEvent(
    new CustomEvent('app-notification', {
      detail: {
        heading: type === 'error' ? 'Notification' : 'Saved',
        message:
          message ||
          (type === 'error' ? 'Something went wrong while saving feedback.' : 'Feedback saved.'),
        type,
      },
    }),
  )
}

function feedbackKey(queryPatchId, resultPatchId) {
  if (queryPatchId == null || resultPatchId == null) return null
  return `${queryPatchId}:${resultPatchId}`
}

function currentFeedbackKey() {
  return feedbackKey(selectedPatch.value?.id, bestMatch.value?.id)
}

function patchHasFeedbackWithSelected(patch) {
  const key = feedbackKey(selectedPatch.value?.id, patch?.id)
  return Boolean(key && feedbackByPair.value[key])
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

    const selectedId = selectedPatch.value?.id
    const selectedName = selectedPatch.value?.patch_filename || patchRouteParam(selectedPatch.value)

    if (
      selectedId != null &&
      !annotatedPatchIds.value.some((id) => String(id) === String(selectedId))
    ) {
      annotatedPatchIds.value = [...annotatedPatchIds.value, selectedId]
    }

    if (selectedName && !annotatedPatchNames.value.includes(selectedName)) {
      annotatedPatchNames.value = [...annotatedPatchNames.value, selectedName]
    }

    applyFeedbackState(savedFeedback)
    showFeedbackToast(label === 'similar' ? 'Marked as similar.' : 'Marked as not similar.')
  } catch (error) {
    feedbackError.value = error.message || 'Failed to save feedback.'
    showFeedbackToast(feedbackError.value, 'error')
  } finally {
    feedbackSaving.value = false
  }
}

async function loadAnnotatedPatches() {
  try {
    const feedbackItems = await fetchMyFeedback()
    const ids = new Set()
    const names = new Set()
    const feedbackPairs = {}

    for (const item of Array.isArray(feedbackItems) ? feedbackItems : []) {
      if (item?.query_patch_id != null) ids.add(String(item.query_patch_id))
      if (item?.query_patch_file_name) names.add(item.query_patch_file_name)

      const key = feedbackKey(item?.query_patch_id, item?.result_patch_id)
      if (key) feedbackPairs[key] = item
    }

    feedbackByPair.value = {
      ...feedbackByPair.value,
      ...feedbackPairs,
    }
    annotatedPatchIds.value = [...ids]
    annotatedPatchNames.value = [...names]
  } catch (error) {
    console.error('Error loading annotated patches:', error)
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
    const results = await fetchSimilarPatches(path, {
      topK: 5,
      sourceImageId: patch.source_image_id,
    })
    const resolvedPatches = (
      await Promise.all(
        results.map(async (item) => {
          try {
            const patchRecord = await fetchPatchByFileName(item.patch_filename)
            return {
              id: patchRecord.id,
              label: item.patch_filename,
              score: (item.similarity_score * 100).toFixed(2),
              imageSrc: getPatchFileUrlByName(item.patch_filename),
              filePath: patchRecord.file_path,
              heatmapSrc: null,
            }
          } catch {
            return null
          }
        }),
      )
    ).filter(Boolean)

    similarPatches.value = resolvedPatches
    await loadExistingFeedbackForCurrentPair()
    loadHeatmapsInBackground(path, resolvedPatches)
  } catch (error) {
    similarError.value = error.message || 'Failed to load similar patches.'
  } finally {
    similarLoading.value = false
  }
}

function loadHeatmapsInBackground(queryPath, patches) {
  patches.forEach(async (patch) => {
    if (!patch.filePath) return
    try {
      const data = await fetchExplainPair(queryPath, patch.filePath)
      const heatmapSrc = getHeatmapFileUrl(data.heatmaps.result)
      const idx = similarPatches.value.findIndex((p) => p.id === patch.id)
      if (idx !== -1) {
        similarPatches.value = [
          ...similarPatches.value.slice(0, idx),
          { ...similarPatches.value[idx], heatmapSrc },
          ...similarPatches.value.slice(idx + 1),
        ]
      }
    } catch {
      // heatmap generation is non-critical — silently skip
    }
  })
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
    const routeValue = decodeURIComponent(fileName.value)
    const data = await fetchImages()
    const allImages = Array.isArray(data) ? data : []

    const foundImage = allImages.find((img) => img.fileName === routeValue)
    let initialSelectedPatchId = null

    if (foundImage) {
      image.value = foundImage
    } else {
      const routePatch = await fetchPatchByFileName(routeValue)
      image.value = allImages.find((img) => img.id === routePatch.source_image_id) || null
      initialSelectedPatchId = routePatch.id
    }

    if (!image.value?.id) return

    const patchData = await fetchPatchesByImageId(image.value.id)

    patches.value = Array.isArray(patchData) ? patchData.map(normalizePatch) : []

    selectedPatch.value = initialSelectedPatchId
      ? patches.value.find((patch) => patch.id === initialSelectedPatchId) ||
        patches.value[0] ||
        null
      : patches.value[0] || null

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
onMounted(loadAnnotatedPatches)

watch(() => route.params.fileName, loadImageFromRoute)
</script>
