<template>
  <div class="min-h-screen bg-[#f7f3ef] text-[#6c4f3d]">
    <main class="mx-auto w-full max-w-[1560px] px-6 py-6">
      <div
        v-if="image"
        class="rounded-[22px] border border-[#cfc5bc] bg-[#fbf8f5] p-6"
      >
        <!-- Top bar -->
        <div class="mb-5 flex items-center justify-between">
          <div class="flex items-center gap-4">
            <button
              type="button"
              class="text-[28px] leading-none"
              @click="router.push('/browse')"
            >
              ←
            </button>

            <span class="text-[17px]">Browse</span>

            <button
              type="button"
              class="ml-6 rounded-full border border-[#8a6755] px-5 py-2 text-[15px]"
            >
              Save
            </button>
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

        <!-- Main content -->
        <div class="grid grid-cols-[180px_minmax(0,1fr)] gap-10">
          <!-- Left image column -->
          <div>
            <div class="overflow-hidden rounded-[18px] bg-[#ebe3db]">
              <img
                :src="mainImageSrc"
                :alt="image.fileName"
                class="h-[640px] w-full object-cover"
                @error="handleMainImageError"
              />
            </div>

            <div class="mt-3 flex items-center justify-center gap-2 text-[14px]">
              <span>{{ displayTitle }}</span>
              <span
                class="flex h-6 w-6 items-center justify-center rounded-full border border-[#a58f82] text-[12px]"
              >
                i
              </span>
            </div>
          </div>

          <!-- Right content column -->
          <div>
            <div class="grid grid-cols-[340px_minmax(0,1fr)] gap-9">
              <!-- Selected patch -->
              <div>
                <p class="mb-3 text-[13px] font-semibold">Selected Patch</p>

                <div class="overflow-hidden rounded-[16px] bg-[#ebe3db]">
                  <img
                    :src="selectedPatchImage"
                    alt="Selected patch"
                    class="h-[265px] w-full object-cover"
                    @error="handleSelectedPatchError"
                  />
                </div>
              </div>

              <!-- Patch info -->
              <div class="pt-1">
                <div>
                  <p class="text-[14px] font-semibold">Patch</p>
                  <h1 class="text-[18px] leading-tight">
                    {{ displayTitle }}
                  </h1>
                </div>

                <div class="mt-14 grid grid-cols-3 gap-8">
                  <div>
                    <p class="text-[14px] font-semibold">Source</p>
                    <p class="text-[18px]">Database</p>
                  </div>

                  <div>
                    <p class="text-[14px] font-semibold">Annotators</p>
                    <p class="text-[18px]">6</p>
                  </div>

                  <div>
                    <p class="text-[14px] font-semibold">Date Uploaded</p>
                    <p class="text-[18px]">--------</p>
                  </div>
                </div>
              </div>
            </div>

            <!-- Similar patches -->
            <div class="mt-8 border-t border-[#d8cec5] pt-4">
              <div class="mb-4 flex items-center justify-between">
                <div class="flex items-center gap-2">
                  <h2 class="text-[16px]">Most similar patches</h2>
                  <span
                    class="flex h-6 w-6 items-center justify-center rounded-full border border-[#a58f82] text-[12px]"
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

              <div class="grid grid-cols-4 gap-6">
                <div
                  v-for="patch in similarPatches"
                  :key="patch.id"
                  class="text-center"
                >
                  <div class="overflow-hidden rounded-[14px] bg-[#ebe3db]">
                    <img
                      :src="patch.imageSrc"
                      :alt="patch.label"
                      class="h-[190px] w-full object-cover"
                      @error="handleSimilarPatchError"
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
            </div>
          </div>
        </div>

        <!-- Bottom action -->
        <div class="mt-10 flex justify-end">
          <button
            type="button"
            class="rounded-full border border-[#8a6755] px-6 py-3 text-[15px]"
          >
            Get random Pen Flourish ›
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
import { apiUrl } from '@/lib/api'

const route = useRoute()
const router = useRouter()

const image = ref(null)
const loading = ref(false)
const heatmapOn = ref(true)
const patches = ref([])

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

const patchTitle = computed(() => {
  const firstPatch = patches.value[0]
  if (firstPatch?.label) return firstPatch.label
  if (!displayTitle.value) return ''
  return `${displayTitle.value}_016`
})

const mainImageSrc = computed(() => {
  if (!image.value?.id) return fallbackImage
  return apiUrl(`/images/${image.value.id}/file`)
})

const selectedPatchImage = computed(() => {
  const firstPatch = patches.value[0]

  if (firstPatch?.imageUrl) return firstPatch.imageUrl
  if (firstPatch?.fileUrl) return firstPatch.fileUrl
  if (firstPatch?.id) return apiUrl(`/patches/${firstPatch.id}/file`)
  if (image.value?.id) return apiUrl(`/images/${image.value.id}/file`)

  return fallbackImage
})

const similarPatches = computed(() => {
  return patches.value.slice(0, 4).map((patch, index) => ({
    ...patch,
    score: [76, 64, 69, 75][index] ?? 70,
    label:
      patch.label ??
      `${displayTitle.value}_${String(index + 11).padStart(3, '0')}`,
    imageSrc:
      patch.imageUrl ||
      patch.fileUrl ||
      (patch.id ? apiUrl(`/patches/${patch.id}/file`) : fallbackImage),
  }))
})

function handleMainImageError(event) {
  event.target.src = fallbackImage
}

function handleSelectedPatchError(event) {
  event.target.src = fallbackImage
}

function handleSimilarPatchError(event) {
  event.target.src = fallbackImage
}

async function loadImageFromRoute() {
  loading.value = true
  image.value = null
  patches.value = []

  try {
    const res = await fetch(apiUrl('/images'))
    if (!res.ok) throw new Error('Failed to load images')

    const data = await res.json()
    const allImages = Array.isArray(data) ? data : []

    const found = allImages.find(
      (img) => img.fileName === decodeURIComponent(fileName.value)
    )

    image.value = found || null

    if (image.value?.id) {
      const patchRes = await fetch(apiUrl(`/images/${image.value.id}/patches`))

      if (patchRes.ok) {
        const patchData = await patchRes.json()
        patches.value = Array.isArray(patchData) ? patchData : []
        console.log('Loaded patches:', patches.value)
      }
    }
  } catch (error) {
    console.error('Error loading image/patches:', error)
  } finally {
    loading.value = false
  }
}

onMounted(loadImageFromRoute)
watch(() => route.params.fileName, loadImageFromRoute)
</script>