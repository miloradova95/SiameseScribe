<template>
  <div
    ref="imageWrapRef"
    class="relative h-[610px] overflow-hidden rounded-[14px] bg-[#ebe3db]"
  >
    <img
      ref="mainImgRef"
      :src="imageSrc"
      class="h-full w-full object-contain"
      @load="updateImageDisplaySize"
      @error="onError"
    />

    <button
      v-for="patch in patches"
      :key="patch.id || patch.patch_filename"
      type="button"
      class="absolute box-border border border-white/80 transition"
      :class="[
        patchIsAnnotated(patch) ? 'bg-[#3f9b4f]/45 ring-1 ring-inset ring-[#2f7d3d]' : 'bg-white/5',
        'hover:z-50 hover:outline hover:outline-2 hover:outline-[#b600ff] hover:outline-offset-[-1px]',
        patchIsAnnotated(patch) ? 'hover:bg-[#3f9b4f]/55' : 'hover:bg-[#b600ff]/10',
        selectedPatchId === patch.id
          ? 'z-50 bg-[#b600ff]/20 outline outline-2 outline-[#b600ff] outline-offset-[-1px]'
          : 'z-10'
      ]"
      :style="patchBoxStyle(patch)"
      @click="$emit('select', patch)"
    />
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  imageSrc: String,
  patches: Array,
  annotatedPatchIds: {
    type: Array,
    default: () => [],
  },
  annotatedPatchNames: {
    type: Array,
    default: () => [],
  },
  selectedPatchId: [String, Number],
})

defineEmits(['select'])

const imageWrapRef = ref(null)
const mainImgRef = ref(null)

const displaySize = ref({
  renderedWidth: 0,
  renderedHeight: 0,
  offsetX: 0,
  offsetY: 0,
  naturalWidth: 1,
  naturalHeight: 1,
})

function onError(e) {
  e.target.src =
    'data:image/svg+xml;charset=UTF-8,' +
    encodeURIComponent(`
      <svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">
        <rect width="100%" height="100%" fill="#ebe3db"/>
      </svg>
    `)
}

function updateImageDisplaySize() {
  const img = mainImgRef.value
  const wrap = imageWrapRef.value
  if (!img || !wrap) return

  const wrapW = wrap.clientWidth
  const wrapH = wrap.clientHeight
  const naturalW = img.naturalWidth || 1
  const naturalH = img.naturalHeight || 1

  const imageRatio = naturalW / naturalH
  const wrapRatio = wrapW / wrapH

  let renderedWidth = wrapW
  let renderedHeight = wrapH
  let offsetX = 0
  let offsetY = 0

  if (imageRatio > wrapRatio) {
    renderedHeight = wrapW / imageRatio
    offsetY = (wrapH - renderedHeight) / 2
  } else {
    renderedWidth = wrapH * imageRatio
    offsetX = (wrapW - renderedWidth) / 2
  }

  displaySize.value = {
    renderedWidth,
    renderedHeight,
    offsetX,
    offsetY,
    naturalWidth: naturalW,
    naturalHeight: naturalH,
  }
}

function patchBoxStyle(patch) {
  const scaleX = displaySize.value.renderedWidth / displaySize.value.naturalWidth
  const scaleY = displaySize.value.renderedHeight / displaySize.value.naturalHeight

  const x = Number(patch.x ?? patch.bbox?.x ?? 0)
  const y = Number(patch.y ?? patch.bbox?.y ?? 0)
  const w = Number(patch.width ?? patch.bbox?.width ?? 128)
  const h = Number(patch.height ?? patch.bbox?.height ?? 128)

  return {
    left: `${Math.round(displaySize.value.offsetX + x * scaleX)}px`,
    top: `${Math.round(displaySize.value.offsetY + y * scaleY)}px`,
    width: `${Math.round(w * scaleX)}px`,
    height: `${Math.round(h * scaleY)}px`,
  }
}

function patchIsAnnotated(patch) {
  const patchId = patch?.id != null ? String(patch.id) : null
  const patchName = patch?.patch_filename || patch?.label || null

  const idMatch =
    patchId != null &&
    Array.isArray(props.annotatedPatchIds) &&
    props.annotatedPatchIds.some((id) => String(id) === patchId)

  const nameMatch =
    patchName != null &&
    Array.isArray(props.annotatedPatchNames) &&
    props.annotatedPatchNames.includes(patchName)

  return idMatch || nameMatch
}
</script>
