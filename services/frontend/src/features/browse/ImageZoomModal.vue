<template>
  <Teleport to="body">
    <transition
      enter-active-class="transition duration-200 ease-out"
      enter-from-class="opacity-0"
      enter-to-class="opacity-100"
      leave-active-class="transition duration-150 ease-in"
      leave-from-class="opacity-100"
      leave-to-class="opacity-0"
    >
      <div
        v-if="open && image"
        class="fixed inset-0 z-[100] bg-black/55 backdrop-blur-[2px]"
        @click.self="closeModal"
      >
        <div class="relative flex h-full w-full items-center justify-center px-6 py-10">
          <button
            type="button"
            class="absolute right-6 top-6 text-white/90 transition hover:opacity-70"
            @click="closeModal"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              class="h-10 w-10"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              stroke-width="1.8"
            >
              <path stroke-linecap="round" stroke-linejoin="round" d="M6 6l12 12M18 6L6 18" />
            </svg>
          </button>

          <div class="relative flex max-h-full max-w-full items-center justify-center">
            <img
              :src="apiUrl(`/images/${image.id}/file`)"
              :alt="image.fileName"
              class="max-h-[82vh] max-w-[80vw] rounded-[10px] object-contain shadow-2xl"
              :style="{ transform: `scale(${zoom})` }"
            />

            <div class="absolute bottom-4 right-[-60px] flex flex-col gap-2">
              <button
                type="button"
                class="flex h-10 w-10 items-center justify-center rounded-lg bg-white/90 text-[#5a4032] shadow transition hover:bg-white"
                @click.stop="zoomIn"
              >
                +
              </button>

              <button
                type="button"
                class="flex h-10 w-10 items-center justify-center rounded-lg bg-white/90 text-[#5a4032] shadow transition hover:bg-white"
                @click.stop="zoomOut"
              >
                −
              </button>
            </div>
          </div>
        </div>
      </div>
    </transition>
  </Teleport>
</template>

<script setup>
import { ref, watch } from 'vue'
import { apiUrl } from '@/lib/api'

const props = defineProps({
  open: {
    type: Boolean,
    default: false,
  },
  image: {
    type: Object,
    default: null,
  },
})

const emit = defineEmits(['close'])

const zoom = ref(1)

watch(
  () => props.open,
  (isOpen) => {
    if (isOpen) zoom.value = 1
  }
)

function closeModal() {
  emit('close')
}

function zoomIn() {
  zoom.value = Math.min(zoom.value + 0.2, 3)
}

function zoomOut() {
  zoom.value = Math.max(zoom.value - 0.2, 0.6)
}
</script>