<template>
  <aside
    class="h-fit rounded-[32px] border border-[#ddd3ca] bg-[#fcfaf8] p-7 shadow-[0_4px_20px_rgba(80,55,35,0.04)]"
  >
    <template v-if="image">
      <div class="mb-6 flex justify-end">
        <button
          type="button"
          class="rounded-full border border-[#8a6755] px-6 py-2 text-[15px] text-[#6c4f3d] transition hover:bg-[#f4ede7]"
        >
          Save
        </button>
      </div>

      <div class="overflow-hidden rounded-[28px] bg-[#ebe3db]">
        <img
          :src="apiUrl(`/images/${image.id}/file`)"
          :alt="image.fileName"
          class="h-[620px] w-full object-cover"
        />
      </div>

      <div class="mt-8 space-y-7">
        <div>
          <p class="text-[12px] font-semibold uppercase tracking-[0.16em] text-[#a08d80]">
            Name
          </p>
          <h2 class="mt-2 break-all text-[28px] font-semibold leading-tight text-[#6c4f3d]">
            {{ image.fileName }}
          </h2>
        </div>

        <div>
          <p class="text-[12px] font-semibold uppercase tracking-[0.16em] text-[#a08d80]">
            Group
          </p>
          <p class="mt-2 text-[22px] leading-tight text-[#6c4f3d]">
            {{ image.group || '—' }}
          </p>
        </div>

        <div>
          <p class="text-[12px] font-semibold uppercase tracking-[0.16em] text-[#a08d80]">
            Source
          </p>
          <div class="mt-2 flex items-center gap-3">
            <p class="text-[22px] leading-tight text-[#6c4f3d]">
              Database
            </p>
            <span
              class="flex h-8 w-8 items-center justify-center rounded-full border border-[#b39d90] text-sm text-[#6c4f3d]"
            >
              i
            </span>
          </div>
        </div>
      </div>

      <div class="mt-8 border-t border-[#e2d8cf] pt-6">
        <div class="flex items-center justify-between gap-4">
         <button
            type="button"
            class="rounded-full border border-[#8a6755] px-5 py-3 text-[15px] text-[#6c4f3d] transition hover:bg-[#f4ede7]"
            @click="emit('zoom')"
          >
            ⊕ Zoom
          </button>

          <button
            type="button"
            class="rounded-full bg-[#7b5a49] px-6 py-3 text-[15px] text-white transition hover:opacity-90"
            @click="goToAnnotate"

          >
            Annotate ›
          </button>
        </div>
      </div>
    </template>

    <template v-else>
      <div class="flex min-h-[520px] items-center justify-center text-center text-[#a49285]">
        Select one image.
      </div>
    </template>
  </aside>
</template>

<script setup>
import { useRouter } from 'vue-router'
import { apiUrl } from '@/lib/api'

const props = defineProps({
  image: {
    type: Object,
    default: null,
  },
})

const emit = defineEmits(['zoom'])
const router = useRouter()

function goToAnnotate() {
  if (!props.image?.fileName) return
  router.push(`/browse/${encodeURIComponent(props.image.fileName)}`)
}
</script>