import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '../stores/auth'
import LoginPage from '@/pages/LoginPage.vue'
import SignupPage from '@/pages/SignupPage.vue'
import HelpPage from '@/pages/HelpPage.vue'
import UploadPage from '@/pages/UploadPage.vue'
import ProfilePage from '@/pages/ProfilePage.vue'
import BrowsePage from '@/pages/BrowsePage.vue'
import AdminPage from '@/pages/AdminPage.vue'
import AnnotatePage from '@/pages/AnnotatePage.vue'


const routes = [
  { path: '/', redirect: '/login' },
  { path: '/login', name: 'login', component: LoginPage },
  { path: '/signup', name: 'signup', component: SignupPage },
  { path: '/upload', name: 'upload', component: UploadPage, meta: { requiresAuth: true } },
  { path: '/profile', name: 'profile', component: ProfilePage, meta: { requiresAuth: true } },
  { path: '/browse', name: 'browse', component: BrowsePage, meta: { requiresAuth: true } },
  { path: '/admin', name: 'admin', component: AdminPage, meta: { requiresAuth: true, adminOnly: true } },
  { path: '/help', name: 'help', component: HelpPage, meta: { requiresAuth: true } },
  { path: '/browse/:fileName', name: 'annotate', component: AnnotatePage, props: true,}
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach(async (to, _from, next) => {
  const authStore = useAuthStore()

  if (authStore.token && !authStore.user) {
    await authStore.fetchCurrentUser()
  }

  if (to.meta.requiresAuth && !authStore.isLoggedIn) {
    return next('/login')
  }
  if (to.meta.adminOnly && !authStore.isAdmin) {
    return next('/browse')
  }
  if (to.path === '/login' && authStore.isLoggedIn) {
    return next('/browse')
  }

  next()
})

export default router
