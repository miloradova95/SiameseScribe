<template>
  <div class="help">
    <section class="search-section">
      <div class="search-bar">
        <input
          v-model="fileName"
          type="text"
          placeholder="Enter patch file name…"
          @keyup.enter="search"
        />
        <button @click="search" :disabled="searching || !fileName.trim()">
          {{ searching ? 'Searching…' : 'Search' }}
        </button>
      </div>
      <p v-if="searchError" class="msg error">{{ searchError }}</p>
    </section>

    <template v-if="queryPatch">
      <section class="results-layout">
        <div class="query-panel">
          <p class="section-label">Query patch</p>
          <div class="patch-card query-card">
            <img
              :src="fileUrl(queryPatch.file_path)"
              :alt="patchName(queryPatch.file_path)"
              class="patch-img"
            />
            <div class="patch-meta">
              <span class="patch-name">{{ patchName(queryPatch.file_path) }}</span>
              <span v-if="queryPatch.group" class="badge">{{ queryPatch.group }}</span>
            </div>
          </div>
        </div>

        <div class="similar-panel">
          <p class="section-label">
            Most similar patches
            <span v-if="embedError" class="msg error" style="font-size: 12px; margin-left: 8px">{{
              embedError
            }}</span>
          </p>

          <div v-if="loadingSimilar" class="msg muted">Finding similar patches…</div>

          <div v-else-if="similarPatches.length" class="similar-grid">
            <div v-for="item in similarPatches" :key="item.patch_filename" class="patch-card">
              <div class="similarity-badge">{{ (item.similarity_score * 100).toFixed(1) }}%</div>
              <img
                :src="patchFileUrl(item.patch_filename)"
                :alt="item.patch_filename"
                class="patch-img"
                @error="onImgError($event)"
              />
              <div class="patch-meta">
                <span class="patch-name">{{ item.patch_filename }}</span>
              </div>
            </div>
          </div>

          <div v-else-if="!loadingSimilar && !embedError" class="msg muted">
            No similar patches found.
          </div>
        </div>
      </section>
    </template>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { apiUrl } from '../lib/api'

const ML_API = import.meta.env.VITE_ML_API_URL ?? 'http://localhost:8001'

const fileName = ref('')
const searching = ref(false)
const searchError = ref(null)
const queryPatch = ref(null)

const loadingSimilar = ref(false)
const embedError = ref(null)
const similarPatches = ref([])

function patchName(filePath) {
  return filePath?.split(/[\\/]/).pop() ?? filePath
}

function fileUrl(filePath) {
  // Serve the patch image via the backend's /patches/{id}/file endpoint.
  // filePath is a relative path stored in the DB; we use the patch id from the object.
  if (!queryPatch.value?.id) return ''
  return apiUrl(`/patches/${queryPatch.value.id}/file`)
}

function patchFileUrl(filename) {
  return apiUrl(`/patches/by-file-name/${encodeURIComponent(filename)}/file`)
}

function onImgError(e) {
  e.target.style.opacity = '0.2'
}

async function search() {
  const name = fileName.value.trim()
  if (!name) return

  searching.value = true
  searchError.value = null
  queryPatch.value = null
  similarPatches.value = []
  embedError.value = null

  try {
    const res = await fetch(apiUrl(`/patches/by-file-name/${encodeURIComponent(name)}`))
    if (!res.ok) throw new Error(`Patch not found (${res.status})`)
    queryPatch.value = await res.json()
  } catch (e) {
    searchError.value = e.message
    searching.value = false
    return
  }

  searching.value = false
  await fetchSimilar(queryPatch.value.file_path)
}

async function fetchSimilar(filePath) {
  loadingSimilar.value = true
  embedError.value = null

  try {
    const embedRes = await fetch(`${ML_API}/embed_patches`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patch_paths: [filePath] }),
    })
    if (!embedRes.ok) throw new Error(`Embedding failed (${embedRes.status})`)
    const embedData = await embedRes.json()
    const vector = embedData.embeddings?.[0]?.vector
    if (!vector) throw new Error('No embedding returned')

    const searchRes = await fetch(`${ML_API}/search_patches`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ embedding: vector, top_k: 8 }),
    })
    if (!searchRes.ok) throw new Error(`Search failed (${searchRes.status})`)
    const searchData = await searchRes.json()

    // Exclude the query patch itself from results
    const queryFile = patchName(filePath)
    similarPatches.value = (searchData.results ?? []).filter((r) => r.patch_filename !== queryFile)
  } catch (e) {
    embedError.value = e.message
  } finally {
    loadingSimilar.value = false
  }
}
</script>

<style scoped>
.help {
  padding: 32px 24px;
  max-width: 1100px;
  margin: 0 auto;
}

.search-section {
  margin-bottom: 40px;
}

.search-bar {
  display: flex;
  gap: 8px;
}

.search-bar input {
  flex: 1;
  padding: 10px 14px;
  font-size: 14px;
  border: 0.5px solid var(--color-border-secondary);
  border-radius: var(--border-radius-md);
  background: var(--color-background-primary);
  color: var(--color-text-primary);
  outline: none;
  transition: border-color 0.15s;
}

.search-bar input:focus {
  border-color: var(--color-border-primary);
  box-shadow: 0 0 0 2px var(--color-border-tertiary);
}

.search-bar button {
  padding: 10px 20px;
  font-size: 14px;
  font-weight: 500;
  background: var(--color-background-primary);
  color: var(--color-text-primary);
  border: 0.5px solid var(--color-border-secondary);
  border-radius: var(--border-radius-md);
  cursor: pointer;
  transition: background 0.15s;
}

.search-bar button:hover:not(:disabled) {
  background: var(--color-background-secondary);
}

.search-bar button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.msg {
  margin-top: 8px;
  font-size: 13px;
}

.muted {
  color: var(--color-text-secondary);
}
.error {
  color: var(--color-text-danger);
}

.results-layout {
  display: flex;
  gap: 40px;
  align-items: flex-start;
}

.query-panel {
  flex: 0 0 200px;
}

.similar-panel {
  flex: 1;
  min-width: 0;
}

.section-label {
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--color-text-secondary);
  margin: 0 0 12px;
}

.patch-card {
  position: relative;
  background: var(--color-background-primary);
  border: 0.5px solid var(--color-border-tertiary);
  border-radius: var(--border-radius-lg);
  overflow: hidden;
}

.query-card {
  width: 200px;
}

.patch-img {
  display: block;
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: cover;
  background: var(--color-background-secondary);
}

.patch-meta {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 10px;
  border-top: 0.5px solid var(--color-border-tertiary);
  min-width: 0;
}

.patch-name {
  font-size: 11px;
  color: var(--color-text-secondary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
  min-width: 0;
}

.badge {
  font-size: 11px;
  padding: 2px 7px;
  border-radius: var(--border-radius-md);
  background: var(--color-background-info);
  color: var(--color-text-info);
  white-space: nowrap;
  flex-shrink: 0;
}

.similarity-badge {
  position: absolute;
  top: 6px;
  right: 6px;
  font-size: 11px;
  font-weight: 500;
  background: rgba(0, 0, 0, 0.55);
  color: #fff;
  padding: 2px 7px;
  border-radius: var(--border-radius-md);
  pointer-events: none;
}

.similar-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 12px;
}

@media (max-width: 680px) {
  .results-layout {
    flex-direction: column;
  }
  .query-panel {
    flex: unset;
    width: 100%;
  }
  .query-card {
    width: 100%;
  }
}
</style>
