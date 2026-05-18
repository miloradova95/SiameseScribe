<template>
  <span
    class="annotatedBadge absolute right-2 top-2 z-10 rounded-full border border-[#2f7d3d] bg-[#3f9b4f] font-bold uppercase tracking-wide text-white shadow-sm"
    :class="sizeClass"
  >
    Annotated
  </span>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  size: {
    type: String,
    default: 'sm',
    validator: (value) => ['sm', 'md'].includes(value),
  },
})

const sizeClass = computed(() =>
  props.size === 'md' ? 'px-3 py-1 text-[11px]' : 'px-2.5 py-1 text-[10px]',
)
</script>
