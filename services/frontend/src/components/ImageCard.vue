<template>
  <button
    type="button"
    class="group w-full overflow-hidden rounded-3xl border bg-white/50 text-left shadow-sm transition hover:-translate-y-0.5 hover:shadow-md"
    :class="selected ? 'border-[#6c4d3d] ring-1 ring-[#6c4d3d]' : 'border-brand-line'"
    @click="$emit('click')"
  >
    <div class="aspect-[4/5] overflow-hidden bg-brand-surface">
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
import { getImageFileUrl } from '../services/image-service'

const props = defineProps({
  image: {
    type: Object,
    required: true,
  },
  selected: {
    type: Boolean,
    default: false,
  },
})

defineEmits(['click'])

const fileUrl = computed(() => getImageFileUrl(props.image.id))
</script>
