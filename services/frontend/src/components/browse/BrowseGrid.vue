<template>
  <section>
    <div
      v-if="loading"
      class="rounded-3xl border border-brand-line bg-white/40 px-6 py-10 text-center text-brand-subtle"
    >
      Lade Bilder...
    </div>

    <div
      v-else-if="images.length === 0"
      class="rounded-3xl border border-brand-line bg-white/40 px-6 py-10 text-center text-brand-subtle"
    >
      Keine Bilder gefunden.
    </div>

    <div
      v-else
      class="grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-4"
    >
      <ImageCard
        v-for="img in images"
        :key="img.id"
        :image="img"
        :selected="selectedImageId === img.id"
        @click="handleSelect(img)"
      />
    </div>
  </section>
</template>

<script setup>
import ImageCard from '@/components/ImageCard.vue'

defineProps({
  images: {
    type: Array,
    default: () => [],
  },
  selectedImageId: {
    type: [String, Number, null],
    default: null,
  },
  loading: Boolean,
})

const emit = defineEmits(['select'])

function handleSelect(img) {
  emit('select', img)
}
</script>