<template>
  <button
    type="button"
    :disabled="disabled || loading"
    class="transition hover:opacity-70 disabled:cursor-not-allowed disabled:opacity-40"
    @click="$emit('click')"
  >
    {{ loading ? loadingText : text }}
  </button>
</template>

<script setup>
defineProps({
  loading: {
    type: Boolean,
    default: false,
  },
  disabled: {
    type: Boolean,
    default: false,
  },
  text: {
    type: String,
    default: 'Get Random Image',
  },
  loadingText: {
    type: String,
    default: 'Loading...',
  },
})

defineEmits(['click'])
</script>
