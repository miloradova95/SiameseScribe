<template>
  <aside class="h-fit rounded-[28px] border border-brand-line bg-white/55 p-5 shadow-sm">
    <template v-if="image">
      <div class="overflow-hidden rounded-3xl bg-brand-surface">
        <img
          :src="`http://localhost:8000/images/${image.id}/file`"
          :alt="image.fileName"
          class="max-h-[460px] w-full object-cover"
        />
      </div>

      <div class="mt-5">
        <p class="text-xs font-semibold uppercase tracking-[0.16em] text-brand-subtle">
          Name
        </p>
        <h2 class="mt-1 break-all text-2xl font-semibold text-brand-text">
          {{ image.fileName }}
        </h2>

        <div class="mt-5 space-y-4">
          <div>
            <p class="text-xs font-semibold uppercase tracking-[0.16em] text-brand-subtle">
              Group
            </p>
            <p class="mt-1 text-lg text-brand-text">
              {{ image.group || '—' }}
            </p>
          </div>

          <div>
            <p class="text-xs font-semibold uppercase tracking-[0.16em] text-brand-subtle">
              Source
            </p>
            <p class="mt-1 text-lg text-brand-text">
              Database
            </p>
          </div>
        </div>
      </div>

      <div class="mt-6 border-t border-brand-line pt-5">
        <div class="mb-3 flex items-center justify-between">
          <h3 class="text-base font-semibold text-brand-text">
            First 4 Patches
          </h3>
          <span class="text-xs text-brand-subtle">
            {{ patches.length }} shown
          </span>
        </div>

        <div v-if="patchLoading" class="text-sm text-brand-subtle">
          Lade Patches...
        </div>

        <div v-else-if="patches.length" class="grid grid-cols-2 gap-3">
          <div
            v-for="patch in patches"
            :key="patch.id"
            class="overflow-hidden rounded-2xl border border-brand-line bg-brand-surface"
          >
            <img
              :src="`http://localhost:8000/patches/${patch.id}/file`"
              :alt="`Patch ${patch.id}`"
              class="aspect-square w-full object-cover"
            />
            <div class="px-2 py-2">
              <p class="text-[11px] text-brand-subtle">
                x={{ patch.bbox.x }}, y={{ patch.bbox.y }}
              </p>
            </div>
          </div>
        </div>

        <div v-else class="text-sm text-brand-subtle">
          Keine Patches gefunden.
        </div>
      </div>
    </template>

    <template v-else>
      <div class="flex min-h-[320px] items-center justify-center text-center text-brand-subtle">
        Wähle ein Bild aus dem Grid aus.
      </div>
    </template>
  </aside>
</template>

<script setup>
defineProps({
  image: {
    type: Object,
    default: null,
  },
  patches: {
    type: Array,
    default: () => [],
  },
  patchLoading: Boolean,
})
</script>