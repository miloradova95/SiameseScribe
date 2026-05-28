<template>
  <div class="min-h-screen bg-[#f7f3ef] text-[#6c4f3d]">
    <main class="mx-auto max-w-[1560px] px-8 pb-10 pt-4">
      <div v-if="image" class="mt-6 grid grid-cols-[210px_minmax(0,1fr)] gap-8">
        <div>
          <AnnotationSidebar
            :image="image"
            :image-src="mainImageSrc"
            :patches="patches"
            :annotated-patch-ids="annotatedPatchIds"
            :annotated-patch-names="annotatedPatchNames"
            :selected-patch-id="selectedPatch?.id"
            :title="displayTitle"
            @back="router.push('/browse')"
            @select-patch="selectPatch"
          />
        </div>

        <section class="rounded-[18px] border border-[#cfc5bc] bg-[#fbf8f5] p-8">
          <div class="mb-6 flex items-start justify-between gap-6">
            <div>
              <h1 class="text-[18px] font-semibold">
                Does the Patch Match the Selected One?
                <span
                  class="inline-flex h-5 w-5 items-center justify-center rounded-full border border-[#6c4f3d] text-xs"
                >
                  i
                </span>
              </h1>

              <p class="mt-2 max-w-[760px] text-[13px] leading-snug text-[#9a8678]">
                Consider the shapes, curves, and style of each one. Do they feel like they
                could be from the same hand? To complete this annotation, you will compare
                two patches, one at a time, to your chosen patch.
              </p>
            </div>

            <div class="flex items-center gap-3 text-[13px]">
              <span>Heatmap</span>
              <button
                type="button"
                class="relative h-6 w-11 rounded-full transition"
                :class="heatmapOnBestMatch ? 'bg-[#7b5a49]' : 'bg-[#d6ccc3]'"
                @click="heatmapOnBestMatch = !heatmapOnBestMatch"
              >
                <span
                  class="absolute top-1 h-4 w-4 rounded-full bg-white transition"
                  :class="heatmapOnBestMatch ? 'left-6' : 'left-1'"
                />
              </button>
            </div>
          </div>

          <p class="mb-4 border-b border-[#d8cec5] pb-2 text-[13px] font-semibold">
            Comparison {{ comparisonIndex + 1 }} of {{ comparisonsPerPatch }}
          </p>

          <div class="grid grid-cols-[1fr_1fr_330px] gap-8">
            <div>
              <div
                class="overflow-hidden rounded-[10px] border border-[#d8cec5] bg-[#ebe3db]"
              >
                <div class="bg-[#5d3d2f] px-4 py-2 text-center text-[13px] text-white">
                  Selected Patch
                </div>

                <img
                  :src="selectedPatchImage"
                  alt="Selected patch"
                  class="h-[275px] w-full object-cover"
                  @error="handleImageError"
                />

                <div class="bg-white px-3 py-3 text-center">
                  <p class="text-[13px] font-semibold text-[#3f2d24]">
                    {{ selectedPatchLabel }}
                  </p>
                  <p class="text-[12px] text-[#9a8678]">Rubricator Group: B</p>
                </div>
              </div>
            </div>

            <div>
              <div
                class="overflow-hidden rounded-[10px] border border-[#d8cec5] bg-[#ebe3db]"
              >
                <div class="bg-white px-4 py-2 text-center text-[13px] text-[#6c4f3d]">
                  Match {{ comparisonIndex + 1 }}
                </div>

                <button
                  type="button"
                  class="relative block w-full bg-[#ebe3db]"
                  :disabled="!currentMatch"
                  @click="openPatch(currentMatch)"
                >
                  <AnnotatedBadge
                    v-if="patchHasFeedbackWithSelected(currentMatch)"
                    size="md"
                  />

                  <img
                    :src="
                      heatmapOnBestMatch && currentMatch?.heatmapSrc
                        ? currentMatch.heatmapSrc
                        : currentMatch?.imageSrc || fallbackImage
                    "
                    :alt="currentMatch?.label || 'Current match'"
                    class="h-[275px] w-full object-cover"
                    @error="handleImageError"
                  />
                </button>

                <div class="bg-white px-3 py-3 text-center">
                  <p class="text-[13px] font-semibold text-[#3f2d24]">
                    {{ currentMatch?.label || "-" }}
                  </p>
                </div>
              </div>
            </div>

            <div>
              <ModelFocusPanel
                :disabled="!modelFocusEnabled"
                @draw="heatmapOnBestMatch = false"
                @erase="console.log('erase')"
                @clear="console.log('clear drawing')"
                @confirm="console.log('confirm drawing')"
                @focus-correct="console.log('focus correct')"
                @example="console.log('show example')"
              />
            </div>
          </div>

          <div class="mt-8 border-[#d8cec5] pt-5">
            <div class="flex flex-col items-center">
              <div class="mb-4 flex w-full items-center gap-3">
                <div class="h-px flex-1 bg-[#d8cec5]"></div>
                <div
                  class="flex h-6 w-6 items-center justify-center rounded-full bg-[#6c4f3d] text-xs font-bold text-white"
                >
                  1
                </div>
                <p class="text-[13px] font-semibold">
                  How similar is it to the chosen patch?
                </p>
                <div class="h-px flex-1 bg-[#d8cec5]"></div>
              </div>

              <div class="flex gap-3">
                <button
                  class="min-w-[88px] rounded-[8px] border px-5 py-3 text-sm transition"
                  :class="feedbackButtonClass('similar')"
                  :disabled="feedbackSaving || !canSubmitFeedback"
                  @click="submitFeedback('similar')"
                >
                  <div
                    class="mx-auto mb-1 flex h-6 w-6 items-center justify-center rounded-full bg-[#dfead4] text-[#567b3e]"
                  >
                    ✓
                  </div>
                  Similar
                </button>

                <button
                  class="min-w-[88px] rounded-[8px] border px-5 py-3 text-sm transition"
                  :class="feedbackButtonClass('not_similar')"
                  :disabled="feedbackSaving || !canSubmitFeedback"
                  @click="submitFeedback('not_similar')"
                >
                  <div
                    class="mx-auto mb-1 flex h-6 w-6 items-center justify-center rounded-full bg-[#f6ded8] text-[#c13b2d]"
                  >
                    ×
                  </div>
                  Not Similar
                </button>

                <button
                  class="min-w-[88px] rounded-[8px] border px-5 py-3 text-sm transition"
                  :class="feedbackButtonClass('uncertain')"
                  :disabled="feedbackSaving || !canSubmitFeedback"
                  @click="submitFeedback('uncertain')"
                >
                  <div
                    class="mx-auto mb-1 flex h-6 w-6 items-center justify-center rounded-full bg-[#f2e6cc] text-[#9a7234]"
                  >
                    ?
                  </div>
                  Uncertain
                </button>
              </div>

              <p
                class="mt-4 text-sm"
                :class="feedbackError ? 'text-red-700' : 'text-[#7f6a5c]'"
              >
                {{ feedbackStatusText }}
              </p>
            </div>
          </div>

          <div class="mt-8 flex justify-end">
            <button
              class="rounded-full border border-[#8a6755] bg-[#eadfd7] px-6 py-2 text-sm text-[#6c4f3d] transition hover:bg-[#dccbbf]"
              @click="goToNextPatch"
            >
              → Next Flourish
            </button>
          </div>
        </section>
      </div>

      <div v-else-if="loading" class="py-20 text-center">Loading...</div>
      <div v-else class="py-20 text-center">Image not found.</div>
    </main>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from "vue";
import { useRoute, useRouter } from "vue-router";
import AnnotatedBadge from "@/features/annotate/AnnotatedBadge.vue";
import AnnotationSidebar from "@/features/annotate/AnnotationSidebar.vue";
import ModelFocusPanel from "@/features/annotate/FocusModelPanel.vue";
import { apiUrl } from "@/lib/api";
import { fetchImages } from "@/services/image-service";
import {
  fetchSimilarPatches,
  fetchPatchByFileName,
  fetchMyFeedbackForPair,
  fetchMyFeedback,
  getPatchFileUrlByName,
  fetchPatchesByImageId,
  patchName,
  saveFeedback,
  fetchExplainPair,
  getHeatmapFileUrl,
} from "@/services/patch-service";

const route = useRoute();
const router = useRouter();

const image = ref(null);
const loading = ref(false);

const heatmapOnBestMatch = ref(false);

const patches = ref([]);
const selectedPatch = ref(null);

const similarPatches = ref([]);
const similarLoading = ref(false);
const similarError = ref(null);

const feedbackSaving = ref(false);
const feedbackError = ref(null);
const lastFeedbackLabel = ref(null);
const feedbackSavedAt = ref(0);
const feedbackByPair = ref({});
const annotatedPatchIds = ref([]);
const annotatedPatchNames = ref([]);

const comparisonIndex = ref(0);
const comparisonsPerPatch = 2;
const completedComparisonsForCurrentPatch = ref(0);

const fileName = computed(() => route.params.fileName);

const modelFocusEnabled = computed(() => {
  return completedComparisonsForCurrentPatch.value >= comparisonsPerPatch;
});

const fallbackImage =
  "data:image/svg+xml;charset=UTF-8," +
  encodeURIComponent(`
    <svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">
      <rect width="100%" height="100%" fill="#ebe3db"/>
    </svg>
  `);

const displayTitle = computed(() => {
  if (!image.value?.fileName) return "";
  return image.value.fileName.replace(/\.[^.]+$/, "");
});

const selectedPatchLabel = computed(() => {
  if (selectedPatch.value?.patch_filename) return selectedPatch.value.patch_filename;
  if (selectedPatch.value?.file_path) return patchName(selectedPatch.value.file_path);
  if (selectedPatch.value?.fileName) return selectedPatch.value.fileName;
  if (selectedPatch.value?.label) return selectedPatch.value.label;
  return displayTitle.value;
});

const currentMatch = computed(() => {
  return similarPatches.value[comparisonIndex.value] || null;
});

const selectedPatchIndex = computed(() => {
  if (!selectedPatch.value || !patches.value.length) return -1;
  return patches.value.findIndex((p) => p.id === selectedPatch.value.id);
});

const canSubmitFeedback = computed(() =>
  Boolean(selectedPatch.value?.id && currentMatch.value?.id)
);

const feedbackStatusText = computed(() => {
  if (feedbackSaving.value) return "Saving feedback...";
  if (feedbackError.value) return feedbackError.value;

  if (lastFeedbackLabel.value && feedbackSavedAt.value) {
    if (completedComparisonsForCurrentPatch.value < comparisonsPerPatch) {
      return "Saved. Please annotate the next comparison.";
    }

    return "Both comparisons saved for this patch.";
  }

  if (!canSubmitFeedback.value) {
    return "Select a patch with a valid match to save feedback.";
  }

  return "Click Similar, Not Similar or Uncertain to save feedback.";
});

const mainImageSrc = computed(() => {
  if (!image.value?.id) return fallbackImage;
  return apiUrl(`/images/${image.value.id}/file`);
});

const selectedPatchImage = computed(() => {
  if (!selectedPatch.value) return fallbackImage;

  if (heatmapOnBestMatch.value && currentMatch.value?.queryHeatmapSrc) {
    return currentMatch.value.queryHeatmapSrc;
  }

  if (selectedPatch.value.id) {
    return apiUrl(`/patches/${selectedPatch.value.id}/file`);
  }

  if (selectedPatch.value.patch_filename) {
    return getPatchFileUrlByName(selectedPatch.value.patch_filename);
  }

  if (selectedPatch.value.imageUrl) return selectedPatch.value.imageUrl;
  if (selectedPatch.value.fileUrl) return selectedPatch.value.fileUrl;

  return fallbackImage;
});

async function selectPatch(patch) {
  selectedPatch.value = patch;
  comparisonIndex.value = 0;
  completedComparisonsForCurrentPatch.value = 0;
  heatmapOnBestMatch.value = false;
  clearFeedbackState();
  await loadSimilarForPatch(patch);
}

function goToNextPatch() {
  if (!patches.value.length) return;

  const idx = selectedPatchIndex.value;
  const next = idx < patches.value.length - 1 ? idx + 1 : 0;

  selectPatch(patches.value[next]);
}

function patchRouteParam(patch) {
  return patch?.label || patch?.patch_filename || patchName(getPatchPath(patch));
}

function openPatch(patch) {
  const target = patchRouteParam(patch);
  if (!target) return;
  router.push(`/browse/${encodeURIComponent(target)}`);
}

function getPatchPath(patch) {
  return (
    patch?.file_path || patch?.filePath || patch?.patch_path || patch?.patch_filename
  );
}

function normalizePatch(patch) {
  return {
    ...patch,
    x: Number(patch.bbox?.x ?? patch.x ?? 0),
    y: Number(patch.bbox?.y ?? patch.y ?? 0),
    width: Number(patch.bbox?.width ?? patch.width ?? 128),
    height: Number(patch.bbox?.height ?? patch.height ?? 128),
  };
}

function handleImageError(event) {
  event.target.src = fallbackImage;
}

function clearFeedbackState() {
  feedbackSaving.value = false;
  feedbackError.value = null;
  lastFeedbackLabel.value = null;
  feedbackSavedAt.value = 0;
}

function showFeedbackToast(message, type = "success") {
  window.dispatchEvent(
    new CustomEvent("app-notification", {
      detail: {
        heading: type === "error" ? "Notification" : "Saved",
        message:
          message ||
          (type === "error"
            ? "Something went wrong while saving feedback."
            : "Feedback saved."),
        type,
      },
    })
  );
}

function feedbackKey(queryPatchId, resultPatchId) {
  if (queryPatchId == null || resultPatchId == null) return null;
  return `${queryPatchId}:${resultPatchId}`;
}

function currentFeedbackKey() {
  return feedbackKey(selectedPatch.value?.id, currentMatch.value?.id);
}

function patchHasFeedbackWithSelected(patch) {
  const key = feedbackKey(selectedPatch.value?.id, patch?.id);
  return Boolean(key && feedbackByPair.value[key]);
}

function applyFeedbackState(feedback) {
  lastFeedbackLabel.value = feedback?.label ?? null;
  feedbackSavedAt.value = feedback ? Date.parse(feedback.created_at) || Date.now() : 0;
}

function feedbackButtonClass(label) {
  const isActive = lastFeedbackLabel.value === label && feedbackSavedAt.value;
  const isDisabled = feedbackSaving.value || !canSubmitFeedback.value;

  if (isActive) return "border-[#6c4f3d] bg-white text-[#5b4033]";
  if (isDisabled) return "border-[#c8baae] text-[#b49f91]";

  return "border-[#d8cec5] bg-white text-[#5b4033] hover:border-[#8a6755]";
}

async function submitFeedback(label) {
  if (!canSubmitFeedback.value || feedbackSaving.value) return;

  feedbackSaving.value = true;
  feedbackError.value = null;

  try {
    await saveFeedback({
      queryPatchId: selectedPatch.value.id,
      resultPatchId: currentMatch.value.id,
      label,
    });

    const key = currentFeedbackKey();

    const savedFeedback = {
      query_patch_id: selectedPatch.value.id,
      result_patch_id: currentMatch.value.id,
      label,
      created_at: new Date().toISOString(),
    };

    if (key) {
      feedbackByPair.value = {
        ...feedbackByPair.value,
        [key]: savedFeedback,
      };
    }

    applyFeedbackState(savedFeedback);
    completedComparisonsForCurrentPatch.value += 1;

    if (completedComparisonsForCurrentPatch.value < comparisonsPerPatch) {
      comparisonIndex.value += 1;
      clearFeedbackState();
      await loadExistingFeedbackForCurrentPair();
      showFeedbackToast("Feedback saved. Next comparison loaded.");
      return;
    }

    const selectedId = selectedPatch.value?.id;
    const selectedName =
      selectedPatch.value?.patch_filename || patchRouteParam(selectedPatch.value);

    if (
      selectedId != null &&
      !annotatedPatchIds.value.some((id) => String(id) === String(selectedId))
    ) {
      annotatedPatchIds.value = [...annotatedPatchIds.value, selectedId];
    }

    if (selectedName && !annotatedPatchNames.value.includes(selectedName)) {
      annotatedPatchNames.value = [...annotatedPatchNames.value, selectedName];
    }

    showFeedbackToast("Both comparisons saved.");
  } catch (error) {
    feedbackError.value = error.message || "Failed to save feedback.";
    showFeedbackToast(feedbackError.value, "error");
  } finally {
    feedbackSaving.value = false;
  }
}

async function loadAnnotatedPatches() {
  try {
    const feedbackItems = await fetchMyFeedback();
    const ids = new Set();
    const names = new Set();
    const feedbackPairs = {};

    for (const item of Array.isArray(feedbackItems) ? feedbackItems : []) {
      if (item?.query_patch_id != null) ids.add(String(item.query_patch_id));
      if (item?.query_patch_file_name) names.add(item.query_patch_file_name);

      const key = feedbackKey(item?.query_patch_id, item?.result_patch_id);
      if (key) feedbackPairs[key] = item;
    }

    feedbackByPair.value = {
      ...feedbackByPair.value,
      ...feedbackPairs,
    };

    annotatedPatchIds.value = [...ids];
    annotatedPatchNames.value = [...names];
  } catch (error) {
    console.error("Error loading annotated patches:", error);
  }
}

async function loadExistingFeedbackForCurrentPair() {
  const key = currentFeedbackKey();

  if (!key) {
    applyFeedbackState(null);
    return;
  }

  if (Object.prototype.hasOwnProperty.call(feedbackByPair.value, key)) {
    applyFeedbackState(feedbackByPair.value[key]);
    return;
  }

  try {
    const feedback = await fetchMyFeedbackForPair({
      queryPatchId: selectedPatch.value.id,
      resultPatchId: currentMatch.value.id,
    });

    feedbackByPair.value = {
      ...feedbackByPair.value,
      [key]: feedback,
    };

    feedbackError.value = null;
    applyFeedbackState(feedback);
  } catch (error) {
    feedbackError.value = error.message || "Failed to load existing feedback.";
  }
}

async function loadSimilarForPatch(patch) {
  comparisonIndex.value = 0;
  completedComparisonsForCurrentPatch.value = 0;

  const path = getPatchPath(patch);

  if (!path) {
    similarPatches.value = [];
    similarError.value = "Selected patch has no file path.";
    return;
  }

  similarLoading.value = true;
  similarError.value = null;
  similarPatches.value = [];
  heatmapOnBestMatch.value = false;

  try {
    const results = await fetchSimilarPatches(path, {
      topK: Math.max(5, comparisonsPerPatch),
      sourceImageId: patch.source_image_id,
    });

    const resolvedPatches = (
      await Promise.all(
        results.map(async (item) => {
          try {
            const patchRecord = await fetchPatchByFileName(item.patch_filename);

            return {
              id: patchRecord.id,
              label: item.patch_filename,
              score: (item.similarity_score * 100).toFixed(2),
              imageSrc: getPatchFileUrlByName(item.patch_filename),
              filePath: patchRecord.file_path,
              heatmapSrc: null,
              queryHeatmapSrc: null,
            };
          } catch {
            return null;
          }
        })
      )
    ).filter(Boolean);

    similarPatches.value = resolvedPatches;

    await loadExistingFeedbackForCurrentPair();
    loadHeatmapsInBackground(path, resolvedPatches);
  } catch (error) {
    similarError.value = error.message || "Failed to load similar patches.";
  } finally {
    similarLoading.value = false;
  }
}

function loadHeatmapsInBackground(queryPath, patchesToLoad) {
  patchesToLoad.forEach(async (patch) => {
    if (!patch.filePath) return;

    try {
      const data = await fetchExplainPair(queryPath, patch.filePath);
      const heatmapSrc = getHeatmapFileUrl(data.heatmaps.result);
      const queryHeatmapSrc = getHeatmapFileUrl(data.heatmaps.query);

      const idx = similarPatches.value.findIndex((p) => p.id === patch.id);

      if (idx !== -1) {
        similarPatches.value = [
          ...similarPatches.value.slice(0, idx),
          {
            ...similarPatches.value[idx],
            heatmapSrc,
            queryHeatmapSrc,
          },
          ...similarPatches.value.slice(idx + 1),
        ];
      }
    } catch {
      // heatmap generation is non-critical
    }
  });
}

async function loadImageFromRoute() {
  loading.value = true;
  image.value = null;
  patches.value = [];
  selectedPatch.value = null;
  similarPatches.value = [];
  similarError.value = null;
  comparisonIndex.value = 0;
  completedComparisonsForCurrentPatch.value = 0;
  clearFeedbackState();

  try {
    const routeValue = decodeURIComponent(fileName.value);
    const data = await fetchImages();
    const allImages = Array.isArray(data) ? data : [];

    const foundImage = allImages.find((img) => img.fileName === routeValue);
    let initialSelectedPatchId = null;

    if (foundImage) {
      image.value = foundImage;
    } else {
      const routePatch = await fetchPatchByFileName(routeValue);
      image.value =
        allImages.find((img) => img.id === routePatch.source_image_id) || null;
      initialSelectedPatchId = routePatch.id;
    }

    if (!image.value?.id) return;

    const patchData = await fetchPatchesByImageId(image.value.id);

    patches.value = Array.isArray(patchData) ? patchData.map(normalizePatch) : [];

    selectedPatch.value = initialSelectedPatchId
      ? patches.value.find((patch) => patch.id === initialSelectedPatchId) ||
        patches.value[0] ||
        null
      : patches.value[0] || null;

    if (selectedPatch.value) {
      await loadSimilarForPatch(selectedPatch.value);
    }
  } catch (error) {
    console.error("Error loading image/patches:", error);
    similarError.value = error.message || "Failed to load image.";
  } finally {
    loading.value = false;
  }
}

onMounted(loadImageFromRoute);
onMounted(loadAnnotatedPatches);

watch(() => route.params.fileName, loadImageFromRoute);
</script>
