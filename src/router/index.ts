import { createRouter, createWebHistory } from "vue-router";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      name: "home",
      component: () => import("../views/Home.vue"),
    },
    {
      path: "/image-segmentation1",
      name: "imageSegment1",
      component: () => import("../views/ImageSegmentation1.vue"),
    },
    {
      path: "/image-segmentation2",
      name: "imageSegment2",
      component: () => import("../views/ImageSegmentation2.vue"),
    },
    {
      path: "/devide-to-pieces",
      name: "devideToPieces",
      component: () => import("../views/DevideToPieces.vue"),
    },
    {
      path: "/face-landmarks",
      name: "faceLandmarks",
      component: () => import("../views/FaceLandmarksDetection.vue"),
    },
  ],
});

export default router;
