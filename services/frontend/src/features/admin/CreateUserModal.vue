<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="fixed inset-0 z-50 flex items-center justify-center bg-black/30 px-4"
      @click.self="$emit('close')"
    >
      <div class="w-full max-w-2xl rounded-3xl bg-[#fbfaf8] p-6 shadow-2xl">
        
        <!-- Header -->
        <div class="mb-5 flex items-center justify-between">
          <div>
            <h2 class="text-2xl font-bold text-[#5b4034]">Create Member</h2>
            <p class="text-sm text-[#9a867c]">Add a new user</p>
          </div>

          <button
            class="rounded-full px-3 py-1 text-2xl text-[#9a867c] hover:bg-[#eee5df]"
            @click="$emit('close')"
          >
            ×
          </button>
        </div>

        <!-- Form -->
        <form @submit.prevent="submit" class="grid gap-4">
          
          <input
            v-model="localForm.username"
            placeholder="Username"
            class="w-full rounded-xl border border-[#d8c9c0] px-4 py-3 text-sm outline-none focus:border-[#5b4034]"
          />

          <input
            v-model="localForm.email"
            placeholder="Email"
            class="w-full rounded-xl border border-[#d8c9c0] px-4 py-3 text-sm outline-none focus:border-[#5b4034]"
          />

          <input
            v-model="localForm.password"
            type="password"
            placeholder="Password"
            class="w-full rounded-xl border border-[#d8c9c0] px-4 py-3 text-sm outline-none focus:border-[#5b4034]"
          />

          <select
            v-model="localForm.role"
            class="w-full rounded-xl border border-[#d8c9c0] px-4 py-3 text-sm outline-none focus:border-[#5b4034]"
          >
            <option value="user">user</option>
            <option value="admin">admin</option>
          </select>

          <!-- Actions -->
          <div class="mt-4 flex justify-end gap-3">
            <button
              type="button"
              class="rounded-full border border-[#bba79d] px-5 py-2 text-sm text-[#5b4034] hover:bg-[#eee5df]"
              @click="$emit('close')"
            >
              Cancel
            </button>

            <button
              type="submit"
              class="rounded-full bg-[#5b4034] px-5 py-2 text-sm text-white hover:bg-[#c53114]"
            >
              Create
            </button>
          </div>

        </form>
      </div>
    </div>
  </Teleport>
</template>

<script setup>
import { ref } from 'vue'

const props = defineProps({
  open: Boolean
})

const emit = defineEmits(['close', 'create'])

const localForm = ref({
  username: '',
  email: '',
  password: '',
  role: 'user'
})

function submit() {
  emit('create', { ...localForm.value })
  localForm.value = {
    username: '',
    email: '',
    password: '',
    role: 'user'
  }
}
</script>