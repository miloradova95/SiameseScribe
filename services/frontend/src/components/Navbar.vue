<template>
  <nav class="flex items-center justify-between px-8 py-6 bg-[#f7f1eb] text-[#2b211d]">
    <router-link
      to="/home"
      class="text-sm uppercase tracking-[0.2em] font-semibold"
    >
      Pena Fleu
    </router-link>

    <div class="flex items-center gap-8 text-sm">
      <router-link to="/browse" class="hover:text-[#c53114] transition">
        Browse
      </router-link>
      <router-link to="/upload" class="hover:text-[#c53114] transition">
        Upload
      </router-link>
      <router-link to="/profile" class="hover:text-[#c53114] transition">
        Profile
      </router-link>
      <router-link to="/home" class="hover:text-[#c53114] transition">
        Home
      </router-link>
      <router-link v-if="authStore.isAdmin" to="/admin" class="hover:text-[#c53114] transition">
        Admin
      </router-link>
      <button
        v-if="authStore.isLoggedIn"
        @click="authStore.logout(); $router.push('/login')"
        class="hover:text-[#c53114] transition"
      >
        Logout
      </button>
    </div>
  </nav>
</template>

<script setup>
import { useAuthStore } from '../stores/auth'
const authStore = useAuthStore()
</script>