<template>
  <div class="min-h-screen bg-[#f7f3ef] text-[#6c4f3d]">
    <main class="mx-auto max-w-[1860px] px-8 pb-10 pt-6">
      <h1 class="font-serif text-[56px] leading-none">
        Browse
      </h1>

      <section class="mt-8 flex gap-5">
        <div class="w-[180px] rounded-[10px] border border-[#c8bbb2] bg-white/20 px-6 py-4">
          <p class="text-[22px] font-bold">{{ totalCount }}</p>
          <p class="text-[11px] text-[#9b8b81]">All</p>
        </div>

        <div class="w-[180px] rounded-[10px] bg-[#e4d9d2] px-6 py-4">
          <p class="text-[22px] font-bold">{{ databaseCount }}</p>
          <p class="text-[11px] text-[#9b8b81]">Database</p>
        </div>

        <div class="w-[180px] rounded-[10px] bg-[#e4d9d2] px-6 py-4">
          <p class="text-[22px] font-bold">{{ userUploadCount }}</p>
          <p class="text-[11px] text-[#9b8b81]">User Uploads</p>
        </div>

        <div class="w-[180px] rounded-[10px] bg-[#e4d9d2] px-6 py-4">
          <p class="text-[22px] font-bold">{{ annotatedCount }}</p>
          <p class="text-[11px] text-[#9b8b81]">Annotated</p>
        </div>
      </section>

      <section class="mt-7 border-b border-[#ddd3ca]">
        <div class="flex gap-8 text-[14px]">
          <button class="pb-3 text-[#a99a90]">Pages</button>

          <button class="border-b border-[#6c4f3d] pb-3 font-semibold">
            Pen Flourishes
          </button>

          <button class="pb-3 text-[#a99a90]">Patches</button>
        </div>
      </section>

      <section class="mt-6 flex items-center justify-between">
        <input
          v-model="searchQuery"
          type="text"
          placeholder="Search by patch or flourish name..."
          class="w-[340px] rounded-full border border-[#cfc2b8] bg-white/30 px-4 py-2 text-[13px] outline-none placeholder:text-[#b4a69d]"
        />

        <div class="flex items-center gap-5">
          <div class="flex items-center gap-3">
            <span class="text-[13px] text-[#8f7d72]">
              Sorted by
            </span>

            <select
              v-model="sortOption"
              class="rounded-full border border-[#d6cbc2] bg-white/30 px-5 py-2 text-[13px] outline-none"
            >
              <option value="most_annotated">Most Annotated</option>
              <option value="least_annotated">Least Annotated</option>
              <option value="name_asc">Name (ASC)</option>
              <option value="name_desc">Name (DESC)</option>
            </select>
          </div>

          <RandomImageButton
            :loading="loading"
            :disabled="filteredImages.length === 0"
            @click="loadRandom"
          />
        </div>
      </section>

      <div
        v-if="error"
        class="mt-6 rounded-2xl border border-[#d6b7a8] bg-[#fff7f2] px-5 py-4 text-sm text-[#a25532]"
      >
        {{ error }}
      </div>

      <section class="mt-6 grid grid-cols-1 items-start gap-10 xl:grid-cols-[minmax(0,1fr)_420px]">
        <BrowseGrid
          :images="paginatedImages"
          :selected-image-id="selectedImage?.id"
          :loading="initialLoading"
          :annotated-image-ids="annotatedImageIds"
          :annotation-counts-by-image-id="annotationCountsByImageId"
          @select="selectImage"
        />

        <BrowseInspector
          :image="selectedImage"
          :patches="patches"
          :patch-loading="patchLoading"
          @zoom="openZoom"
        />
      </section>

      <Pagination
        :current-page="currentPage"
        :total-pages="totalPages"
        :page-size="pageSize"
        @change="goToPage"
      />
    </main>

    <Teleport to="body">
      <div
        v-if="zoomOpen && selectedImage"
        class="fixed inset-0 z-[100] bg-black/55 backdrop-blur-[2px]"
        @click.self="closeZoom"
      >
        <div class="relative flex h-full w-full items-center justify-center px-6 py-10">
          <button
            type="button"
            class="absolute right-6 top-6 text-white/90"
            @click="closeZoom"
          >
            ✕
          </button>

          <img
            :src="apiUrl(`/images/${selectedImage.id}/file`)"
            :alt="selectedImage.fileName"
            class="max-h-[82vh] max-w-[80vw] rounded-[10px] object-contain shadow-2xl"
            :style="{ transform: `scale(${zoomLevel})` }"
          />
        </div>
      </div>
    </Teleport>
  </div>
</template>
<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";

import BrowseGrid from "@/features/browse/components/BrowseGrid.vue";
import BrowseInspector from "@/features/browse/BrowseInspector.vue";
import RandomImageButton from "@/components/RandomImageButton.vue";
import Pagination from "@/features/browse/components/Pagination.vue";

import { apiUrl } from "@/lib/api";
import { fetchImages } from "@/services/image-service";
import { fetchMyFeedback, fetchPatchesByImageId } from "@/services/patch-service";

const images = ref([]);
const selectedImage = ref(null);
const patches = ref([]);

const initialLoading = ref(false);
const loading = ref(false);
const patchLoading = ref(false);
const error = ref(null);

const annotatedImageIds = ref([]);
const annotationCountsByImageId = ref({});

const currentPage = ref(1);
const pageSize = ref(24);
const searchQuery = ref("");
const sortOption = ref("most_annotated");

const zoomOpen = ref(false);
const zoomLevel = ref(1);

let currentPatchRequestId = 0;

const totalCount = computed(() => images.value.length);

const databaseCount = computed(() =>
  images.value.filter((img) => (img.source || "Database") === "Database").length
);

const userUploadCount = computed(() =>
  images.value.filter((img) =>
    ["User Uploads", "User Upload", "Upload"].includes(img.source)
  ).length
);

const annotatedCount = computed(() => annotatedImageIds.value.length);

const filteredImages = computed(() => {
  const query = searchQuery.value.trim().toLowerCase();

  let result = [...images.value];

  // SEARCH
  if (query) {
    result = result.filter((img) => {
      return (
        img.fileName?.toLowerCase().includes(query) ||
        img.name?.toLowerCase().includes(query) ||
        img.group?.toLowerCase().includes(query) ||
        img.source?.toLowerCase().includes(query) ||
        String(img.id).includes(query)
      );
    });
  }

  // SORTING
  switch (sortOption.value) {
    case "most_annotated":
      result.sort((a, b) => {
        const countA =
          annotationCountsByImageId.value[String(a.id)] || 0;

        const countB =
          annotationCountsByImageId.value[String(b.id)] || 0;

        return countB - countA;
      });
      break;

    case "least_annotated":
      result.sort((a, b) => {
        const countA =
          annotationCountsByImageId.value[String(a.id)] || 0;

        const countB =
          annotationCountsByImageId.value[String(b.id)] || 0;

        return countA - countB;
      });
      break;

    case "name_asc":
      result.sort((a, b) =>
        (a.fileName || "").localeCompare(b.fileName || "")
      );
      break;

    case "name_desc":
      result.sort((a, b) =>
        (b.fileName || "").localeCompare(a.fileName || "")
      );
      break;
  }

  return result;
});

const totalPages = computed(() =>
  Math.max(1, Math.ceil(filteredImages.value.length / pageSize.value))
);

const paginatedImages = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value;
  return filteredImages.value.slice(start, start + pageSize.value);
});

watch(searchQuery, () => {
  currentPage.value = 1;

  if (paginatedImages.value.length > 0) {
    selectImage(paginatedImages.value[0]);
  } else {
    selectedImage.value = null;
    patches.value = [];
  }
});

async function loadAllImages() {
  initialLoading.value = true;
  error.value = null;

  try {
    const data = await fetchImages();
    images.value = Array.isArray(data) ? data : [];
    currentPage.value = 1;

    if (paginatedImages.value.length > 0) {
      await selectImage(paginatedImages.value[0]);
    }
  } catch (err) {
    error.value = err.message || "Etwas ist schiefgelaufen.";
  } finally {
    initialLoading.value = false;
  }
}

async function loadAnnotatedImages() {
  try {
    const feedbackItems = await fetchMyFeedback();
    const ids = new Set();
    const counts = {};

    for (const item of Array.isArray(feedbackItems) ? feedbackItems : []) {
      const imageId = item?.query_patch_source_image_id;

      if (imageId != null) {
        const key = String(imageId);
        ids.add(key);
        counts[key] = (counts[key] || 0) + 1;
      }
    }

    annotatedImageIds.value = [...ids];
    annotationCountsByImageId.value = counts;
  } catch (err) {
    console.error("Error loading annotated images:", err);
  }
}

async function loadPatchesForImage(imageId) {
  const requestId = ++currentPatchRequestId;
  patchLoading.value = true;
  patches.value = [];

  try {
    const data = await fetchPatchesByImageId(imageId);

    if (requestId !== currentPatchRequestId) return;

    patches.value = Array.isArray(data) ? data.slice(0, 4) : [];
  } catch (err) {
    if (requestId !== currentPatchRequestId) return;
    error.value = err.message || "Failed to fetch patches.";
  } finally {
    if (requestId === currentPatchRequestId) {
      patchLoading.value = false;
    }
  }
}

async function selectImage(img) {
  if (!img?.id) return;

  selectedImage.value = img;
  error.value = null;

  await loadPatchesForImage(img.id);
}

async function goToPage(page) {
  currentPage.value = Math.min(Math.max(page, 1), totalPages.value);

  if (paginatedImages.value.length > 0) {
    await selectImage(paginatedImages.value[0]);
  }
}

async function loadRandom() {
  if (filteredImages.value.length === 0) return;

  loading.value = true;
  error.value = null;

  try {
    const pool = filteredImages.value;
    const randomIndex = Math.floor(Math.random() * pool.length);
    const randomImage = pool[randomIndex];

    const index = filteredImages.value.findIndex((img) => img.id === randomImage.id);

    if (index !== -1) {
      currentPage.value = Math.floor(index / pageSize.value) + 1;
    }

    await selectImage(randomImage);
  } finally {
    loading.value = false;
  }
}

function openZoom() {
  if (!selectedImage.value) return;

  zoomLevel.value = 1;
  zoomOpen.value = true;
}

function closeZoom() {
  zoomOpen.value = false;
}

function handleKeydown(event) {
  if (event.key === "Escape") {
    closeZoom();
  }
}

onMounted(() => {
  loadAllImages();
  loadAnnotatedImages();
  window.addEventListener("keydown", handleKeydown);
});

onBeforeUnmount(() => {
  window.removeEventListener("keydown", handleKeydown);
});
</script>