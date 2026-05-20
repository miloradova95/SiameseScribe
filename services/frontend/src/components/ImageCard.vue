<template>
  <button
    type="button"
    class="group w-full overflow-hidden rounded-[10px] border bg-[#f8f4ef] text-left transition hover:shadow-md"
    :class="selected ? 'border-[#6c4d3d]' : 'border-[#ddd1c8]'"
    @click="$emit('click')"
  >
    <div class="relative aspect-[1.05/1] overflow-hidden bg-[#e8ded6]">
      <AnnotatedBadge
        v-if="annotated"
        position="top-left"
        class="absolute left-3 top-3 z-10"
      />

      <div class="absolute right-3 top-3 z-10">
        <BookmarkButton :item="image" />
      </div>

      <img
        :src="fileUrl"
        :alt="image.fileName"
        class="h-full w-full object-cover transition duration-300 group-hover:scale-[1.03]"
      />
    </div>

    <div class="px-3 pb-3 pt-2">
      <p class="truncate text-[12px] font-medium text-[#6c4d3d]">
        {{ displayName }}
      </p>

      <p class="mt-1 text-[10px] text-[#9d8c82]">
        Group {{ image.group || "—" }} · {{ image.source || "Database" }}
      </p>

      <p class="mt-0.5 text-[10px] text-[#9d8c82]">{{ annotationCount }} Annotations</p>
    </div>
  </button>
</template>

<script setup>
import { computed } from "vue";
import { getImageFileUrl } from "@/services/image-service";
import AnnotatedBadge from "@/features/annotate/AnnotatedBadge.vue";
import BookmarkButton from "@/components/BookmarkButton.vue";

const props = defineProps({
  image: {
    type: Object,
    required: true,
  },
  selected: Boolean,
  annotated: Boolean,
  annotationCount: {
    type: Number,
    default: 0,
  },
});

defineEmits(["click"]);

const fileUrl = computed(() => getImageFileUrl(props.image.id));

const displayName = computed(() => {
  return props.image.name || props.image.fileName?.replace(/\.[^/.]+$/, "") || "Untitled";
});
</script>
