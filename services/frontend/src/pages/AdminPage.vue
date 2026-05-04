<template>
  <div class="p-8 max-w-6xl mx-auto space-y-10">
      <h1 class="text-2xl font-semibold text-[#2b211d] mb-6">Admin Panel</h1>

      <div class="mb-8 p-4 border border-gray-200 rounded">
        <h2 class="font-semibold mb-3">Create User</h2>
        <form @submit.prevent="createUser" class="flex flex-wrap gap-3 items-end">
          <div>
            <label class="block text-xs mb-1">Username</label>
            <input v-model="form.username" type="text" required class="border rounded px-2 py-1 text-sm" />
          </div>
          <div>
            <label class="block text-xs mb-1">Email</label>
            <input v-model="form.email" type="email" required class="border rounded px-2 py-1 text-sm" />
          </div>
          <div>
            <label class="block text-xs mb-1">Password</label>
            <input v-model="form.password" type="password" required class="border rounded px-2 py-1 text-sm" />
          </div>
          <div>
            <label class="block text-xs mb-1">Role</label>
            <select v-model="form.role" class="border rounded px-2 py-1 text-sm">
              <option value="user">user</option>
              <option value="admin">admin</option>
            </select>
          </div>
          <button type="submit" class="bg-[#2b211d] text-white px-4 py-1 rounded text-sm hover:bg-[#c53114] transition">
            Create
          </button>
        </form>
        <p v-if="createError" class="text-red-600 text-sm mt-2">{{ createError }}</p>
        <p v-if="createSuccess" class="text-green-600 text-sm mt-2">{{ createSuccess }}</p>
      </div>

    <!-- User table -->
    <table class="w-full text-sm border-collapse">
      <thead>
        <tr class="border-b text-left text-xs uppercase text-gray-500">
          <th class="py-2 pr-4">Username</th>
          <th class="py-2 pr-4">Email</th>
          <th class="py-2 pr-4">Role</th>
          <th class="py-2 pr-4">Active</th>
          <th class="py-2"></th>
        </tr>
      </thead>
      <tbody>
      <tr
        v-for="u in users"
        :key="u.id"
        class="border-t border-[#d7cec7] transition hover:bg-[#f4eee9]"
      >
        <td class="px-5 py-4 font-medium text-[#2b211d]">
          {{ u.username }}
        </td>

        <td class="px-5 py-4 text-[#5b4033]">
          {{ u.email }}
        </td>

        <td class="px-5 py-4">
          <span
            class="rounded-full px-3 py-1 text-xs font-medium"
            :class="
              u.role === 'admin'
                ? 'bg-[#6c4f3d] text-white'
                : 'bg-[#e5dbd4] text-[#5b4033]'
            "
          >
            {{ u.role }}
          </span>
        </td>

        <td class="px-5 py-4">
          <span
            class="rounded-full px-3 py-1 text-xs font-medium"
            :class="
              u.is_active
                ? 'bg-[#dcebc9] text-green-800'
                : 'bg-[#efd6cf] text-red-800'
            "
          >
            {{ u.is_active ? 'Active' : 'Inactive' }}
          </span>
        </td>

        <td class="px-5 py-4">
          <div
            v-if="u.id !== authStore.user?.id"
            class="flex justify-end gap-2"
          >
            <button
              v-if="u.is_active"
              @click="deactivate(u.id)"
              class="rounded-full border border-[#c53114] px-3 py-1 text-xs text-[#c53114] transition hover:bg-[#c53114] hover:text-white"
            >
              Deactivate
            </button>

            <button
              v-else
              @click="activate(u.id)"
              class="rounded-full border border-green-700 px-3 py-1 text-xs text-green-700 transition hover:bg-green-700 hover:text-white"
            >
              Activate
            </button>

            <button
              @click="deleteUser(u.id)"
              class="rounded-full bg-[#c53114] px-3 py-1 text-xs text-white transition hover:opacity-80"
            >
              Delete
            </button>
          </div>

          <span
            v-else
            class="block text-right text-xs text-[#9b8b82]"
          >
            Current user
          </span>
        </td>
      </tr>
    </tbody>
    </table>
    <p v-if="loadError" class="text-red-600 text-sm mt-4">{{ loadError }}</p>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useAuthStore } from '../stores/auth'
import { fetchWithAuth, apiUrl } from '../lib/api'
import { fetchAdminFeedback, retrainFromFeedback } from '../services/patch-service'

const authStore = useAuthStore()

const users = ref([])
const loadError = ref('')
const createError = ref('')
const createSuccess = ref('')
const form = ref({ username: '', email: '', password: '', role: 'user' })

const feedbackItems = ref([])
const feedbackLoading = ref(false)
const feedbackError = ref('')
const retrainLoading = ref(false)
const retrainError = ref('')
const retrainSuccess = ref('')
const kTriplets = ref(1)
const selectedFeedbackIds = ref(new Set())
const feedbackFilters = ref({
  userId: '',
  dateFrom: '',
  dateTo: '',
  usedForRetrain: 'false',
})

const selectedCount = computed(() => selectedFeedbackIds.value.size)
const visibleSelectableIds = computed(() => feedbackItems.value.filter((item) => !item.used_for_retrain).map((item) => item.id))
const allVisibleSelected = computed(() => (
  visibleSelectableIds.value.length > 0 &&
  visibleSelectableIds.value.every((id) => selectedFeedbackIds.value.has(id))
))

async function loadUsers() {
  loadError.value = ''
  const res = await fetchWithAuth(apiUrl('/users'))
  if (!res.ok) {
    loadError.value = 'Failed to load users'
    return
  }
  users.value = await res.json()
}

async function createUser() {
  createError.value = ''
  createSuccess.value = ''
  const res = await fetchWithAuth(apiUrl('/users'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(form.value),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    createError.value = err.detail ?? 'Failed to create user'
    return
  }
  createSuccess.value = `User "${form.value.username}" created.`
  form.value = { username: '', email: '', password: '', role: 'user' }
  await loadUsers()
}

async function deactivate(userId) {
  const res = await fetchWithAuth(apiUrl(`/users/${userId}/deactivate`), { method: 'PATCH' })
  if (res.ok) await loadUsers()
}

async function activate(userId) {
  const res = await fetchWithAuth(apiUrl(`/users/${userId}/activate`), {
    method: 'PATCH',
  })

  if (res.ok) await loadUsers()
}

async function deleteUser(userId) {
  if (!confirm('Delete this user permanently?')) return

  const res = await fetchWithAuth(apiUrl(`/users/${userId}`), {
    method: 'DELETE',
  })

  if (res.ok) await loadUsers()
}

onMounted(loadUsers)
</script>
