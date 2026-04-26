<template>
  <section class="w-full">
    <div class="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm">
      <div v-if="!image" class="text-sm text-zinc-500">
        No image selected.
      </div>

      <div v-else class="space-y-4">
        <div class="flex items-center justify-between gap-4">
          <div class="text-left">
            <h2 class="text-lg font-semibold">{{ image.filename }}</h2>
            <p class="text-sm text-zinc-500">
              {{ patches.length }} patches
            </p>
          </div>
        </div>

        <div
          ref="viewerRef"
          class="relative mx-auto w-full overflow-auto rounded-xl border bg-zinc-50"
          style="max-height: 75vh;"
        >
          <img
            ref="imgRef"
            :src="image.image_url"
            :alt="image.filename"
            class="block h-auto w-full select-none"
            @load="handleImageLoad"
          />

          <div
            v-for="patch in visiblePatches"
            :key="patch.source_image_id + '-' + patch.patch_filename"
            class="absolute cursor-pointer border transition"
            :class="isSelected(patch)
              ? 'border-red-500 bg-red-500/20'
              : 'border-sky-500 bg-sky-500/10 hover:bg-sky-500/20'"
            :style="patchStyle(patch)"
            @click="selectPatch(patch)"
            :title="`${patch.patch_filename} (${patch.x}, ${patch.y})`"
          />
        </div>

        <div class="text-left text-sm text-zinc-500">
        </div>
      </div>
    </div>
  </section>
</template>

<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  image: { type: Object, default: null },
  patches: { type: Array, default: () => [] },
  patchLoading: { type: Boolean, default: false },
  selectedPatch: { type: Object, default: null }
})

const emit = defineEmits(['select-patch'])

const imgRef = ref(null)

const renderedWidth = ref(1)
const renderedHeight = ref(1)

const originalWidth = ref(1)
const originalHeight = ref(1)

const PATCH_SIZE = 256

function handleImageLoad() {
  if (!imgRef.value) return

  renderedWidth.value = imgRef.value.clientWidth
  renderedHeight.value = imgRef.value.clientHeight
  originalWidth.value = imgRef.value.naturalWidth
  originalHeight.value = imgRef.value.naturalHeight
}

const scaleX = computed(() => renderedWidth.value / originalWidth.value)
const scaleY = computed(() => renderedHeight.value / originalHeight.value)

function patchStyle(patch) {
  return {
    left: `${patch.x * scaleX.value}px`,
    top: `${patch.y * scaleY.value}px`,
    width: `${PATCH_SIZE * scaleX.value}px`,
    height: `${PATCH_SIZE * scaleY.value}px`
  }
}

function selectPatch(patch) {
  emit('select-patch', patch)
}

function isSelected(patch) {
  return props.selectedPatch?.patch_filename === patch.patch_filename
}


const visiblePatches = computed(() => {
  if (!props.patches?.length) return []

  return props.patches.slice(0, 2000)
})
</script>