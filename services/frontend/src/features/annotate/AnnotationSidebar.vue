<template>
  <aside>
    <div class="mb-5 flex items-center justify-between">
      <button
        class="rounded-lg bg-[#8a6755] px-3 py-1 text-sm font-bold text-white transition hover:bg-[#6c4f3d]"
        @click="$emit('back')"
      >
        ← Back
      </button>
      <span class="text-[17px]">Browse</span>
      <BookmarkButton :item="image" />
    </div>

    <PatchOverlayImage
      :image-src="imageSrc"
      :patches="patches"
      :annotated-patch-ids="annotatedPatchIds"
      :annotated-patch-names="annotatedPatchNames"
      :selected-patch-id="selectedPatchId"
      @select="$emit('select-patch', $event)"
    />

    <div class="mt-3 flex items-center justify-center gap-2 text-[14px]">
      <span>{{ title }}</span>
      <span class="flex h-5 w-5 items-center justify-center rounded-full border text-xs">
        i
      </span>
    </div>
  </aside>
</template>

<script setup>
import PatchOverlayImage from './PatchOverlayImage.vue'
import BookmarkButton from '@/components/BookmarkButton.vue'

defineProps({
  image: Object,
  imageSrc: String,
  patches: Array,
  annotatedPatchIds: Array,
  annotatedPatchNames: Array,
  selectedPatchId: [String, Number],
  title: String,
})

defineEmits(['select-patch', 'back'])
</script>
