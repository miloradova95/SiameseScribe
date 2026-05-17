<template>
  <div>
    <Navbar v-if="route.meta.requiresAuth" />
    <div class="pointer-events-none fixed bottom-6 left-1/2 z-50 flex -translate-x-1/2 flex-col-reverse items-center gap-3">
      <transition-group
        enter-active-class="transition duration-200 ease-out"
        enter-from-class="translate-y-2 opacity-0"
        enter-to-class="translate-y-0 opacity-100"
        leave-active-class="transition duration-150 ease-in"
        leave-from-class="translate-y-0 opacity-100"
        leave-to-class="translate-y-2 opacity-0"
      >
        <PopupNotification
          v-for="item in notifications"
          :key="item.id"
          :message="item.message"
          :type="item.type"
          :heading="item.heading"
        />
      </transition-group>
    </div>
    <router-view />
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'
import Navbar from '@/components/Navbar.vue'
import PopupNotification from '@/components/PopupNotification.vue'
import { useRoute } from 'vue-router'

const route = useRoute()

const notifications = ref([])
const notificationTimeoutIds = new Map()

function removeNotification(id) {
  notifications.value = notifications.value.filter(item => item.id !== id)

  const timeoutId = notificationTimeoutIds.get(id)
  if (timeoutId) {
    window.clearTimeout(timeoutId)
    notificationTimeoutIds.delete(id)
  }
}

function showNotification(detail = {}) {
  const id =
    window.crypto?.randomUUID?.() ||
    `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`

  notifications.value = [
    ...notifications.value,
    {
      id,
      message: detail.message || 'Saved.',
      type: detail.type || 'success',
      heading: detail.heading || (detail.type === 'error' ? 'Notification' : 'Saved'),
    },
  ]

  const timeoutId = window.setTimeout(() => {
    removeNotification(id)
  }, 2600)

  notificationTimeoutIds.set(id, timeoutId)
}

function hideAllNotifications() {
  notifications.value = []

  for (const timeoutId of notificationTimeoutIds.values()) {
    window.clearTimeout(timeoutId)
  }

  notificationTimeoutIds.clear()
}

function handleAppNotification(event) {
  showNotification(event.detail)
}

onMounted(() => {
  window.addEventListener('app-notification', handleAppNotification)
})

onBeforeUnmount(() => {
  window.removeEventListener('app-notification', handleAppNotification)
  hideAllNotifications()
})
</script>
