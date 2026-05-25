<template>
  <button
    type="button"
    class="group w-full overflow-hidden rounded-3xl border bg-white/50 text-left shadow-sm transition hover:-translate-y-0.5 hover:shadow-md"
    :class="selected ? 'border-[#6c4d3d] ring-1 ring-[#6c4d3d]' : 'border-brand-line'"
    @click="$emit('click')"
  >
    <div class="relative aspect-[4/5] overflow-hidden bg-brand-surface">
      
      <!-- ✅ Bookmark Button -->
      <div class="absolute top-2 right-2 z-10">
        <BookmarkButton :item="image" />
      </div>

      <AnnotatedBadge v-if="annotated" position="top-left" />

      <img
        :src="fileUrl"
        :alt="image.fileName"
        class="h-full w-full object-cover transition duration-300 group-hover:scale-[1.02]"
      />
    </div>

    <div class="px-3 py-3">
      <p class="truncate text-sm font-medium text-brand-text">
        {{ image.fileName }}
      </p>
      <p class="mt-1 text-xs text-brand-subtle">
        Group: {{ image.group || '—' }}
      </p>
    </div>
  </button>
</template>

<script setup>
import { computed } from 'vue'
import { getImageFileUrl } from '@/services/image-service'
import AnnotatedBadge from '@/features/annotate/AnnotatedBadge.vue'
import BookmarkButton from '@/components/BookmarkButton.vue'

const props = defineProps({
  image: Object,
  selected: Boolean,
  annotated: Boolean,
})

defineEmits(['click'])

const fileUrl = computed(() => getImageFileUrl(props.image.id))
</script>
