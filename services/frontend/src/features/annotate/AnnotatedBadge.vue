<template>
  <span
    class="annotatedBadge absolute z-10 rounded-full bg-[#FEF6ED] font-semibold tracking-wide text-[#563B2E] shadow-sm"
    :class="[sizeClass, positionClass]"
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
  position: {
    type: String,
    default: 'top-right',
    validator: (value) => ['top-left', 'top-right'].includes(value),
  },
})

const sizeClass = computed(() =>
  props.size === 'md' ? 'px-3 py-2 text-[15px]' : 'px-3 py-2 text-[15px]',
)

const positionClass = computed(() =>
  props.position === 'top-left' ? 'left-2 top-2' : 'right-2 top-2',
)
</script>
