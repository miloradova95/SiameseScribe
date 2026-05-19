<template>
  <div class="relative inline-block">
    <button
      type="button"
      class="flex h-8 w-8 items-center justify-center rounded-full border border-[#b39d90] text-sm text-[#6c4f3d]"
      @click.stop="toggle"
    >
      i
    </button>

    <div
      v-if="open"
      class="absolute z-50 w-56 rounded-2xl border border-[#ddd3ca] bg-[#fcfaf8] p-3 text-sm text-[#6c4f3d] shadow-lg"
      :class="positionClass"
    >
      {{ text }}
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  text: String,
  position: {
    type: String,
    default: 'right', // 'right' | 'left' | 'top' | 'bottom'
  },
})

const open = ref(false)

const positionClass = {
  right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  left: 'right-full top-1/2 -translate-y-1/2 mr-2',
  top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
  bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
}[props.position]

function toggle() {
  open.value = !open.value
}

// Schließen bei Klick außerhalb
function onClickOutside() {
  open.value = false
}

import { onMounted, onUnmounted } from 'vue'
onMounted(() => document.addEventListener('click', onClickOutside))
onUnmounted(() => document.removeEventListener('click', onClickOutside))
</script>