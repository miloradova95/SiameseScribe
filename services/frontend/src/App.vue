<template>
  <div>
  <Navbar v-if="route.meta.requiresAuth" />
    <router-view />
  </div>
</template>

<script setup>
import Navbar from '@/components/Navbar.vue'
import { useRoute } from 'vue-router'
const route = useRoute()
</script>