<template>
  <div class="p-8 max-w-3xl mx-auto">
    <h1 class="text-2xl font-semibold text-[#2b211d] mb-6">Admin — Manage Users</h1>

    <!-- Create user form -->
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
        <tr v-for="u in users" :key="u.id" class="border-b">
          <td class="py-2 pr-4">{{ u.username }}</td>
          <td class="py-2 pr-4">{{ u.email }}</td>
          <td class="py-2 pr-4">{{ u.role }}</td>
          <td class="py-2 pr-4">{{ u.is_active ? '✓' : '✗' }}</td>
          <td class="py-2">
            <button
              v-if="u.is_active && u.id !== authStore.user?.id"
              @click="deactivate(u.id)"
              class="text-red-600 text-xs hover:underline"
            >
              Deactivate
            </button>
          </td>
        </tr>
      </tbody>
    </table>
    <p v-if="loadError" class="text-red-600 text-sm mt-4">{{ loadError }}</p>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useAuthStore } from '../stores/auth'
import { fetchWithAuth, apiUrl } from '../lib/api'

const authStore = useAuthStore()

const users = ref([])
const loadError = ref('')
const createError = ref('')
const createSuccess = ref('')
const form = ref({ username: '', email: '', password: '', role: 'user' })

async function loadUsers() {
  loadError.value = ''
  const res = await fetchWithAuth(apiUrl('/users'))
  if (!res.ok) { loadError.value = 'Failed to load users'; return }
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

onMounted(loadUsers)
</script>
