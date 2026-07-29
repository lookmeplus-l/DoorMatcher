package com.doormatcher.native

import android.graphics.Bitmap

/**
 * 纯 Kotlin 实现的三指标相似度比对（Native 等效）
 * 与 DoorMatcherNative 算法完全兼容，供 TFLite 未加载时 fallback 使用
 */
object DoorMatcherPureKotlin {

    /**
     * 比对结果
     */
    data class MatchResult(
        val filename: String,
        val cosineSimilarity: Float,
        val histogramSimilarity: Float,
        val ssimSimilarity: Float,
        val combinedScore: Float
    )

    // 区域比例（与 DoorDatabase 一致）
    private const val XR1 = 0.25f
    private const val XR2 = 0.75f
    private const val YR1 = 0.25f
    private const val YR2 = 0.75f

    private const val HIST_BINS = 64
    private const val W_COS  = 0.35f
    private const val W_HIST = 0.15f
    private const val W_SSIM = 0.50f

    /**
     * 比对截图与所有门图，返回排序结果
     */
    fun match(bitmap: Bitmap, doorEntries: List<DoorFeatures>): List<MatchResult> {
        val shotRegion = extractRegion(bitmap, XR1, XR2, YR1, YR2, 450, 750)
        val shotHist    = computeHistogram(shotRegion)

        return matchWithFeatures(shotRegion, shotHist, doorEntries)
    }

    /**
     * 用预提取特征直接比对（推荐路径，最快）
     */
    fun matchWithEntries(
        shotBitmap: Bitmap,
        doorEntries: List<DoorFeatures>
    ): List<MatchResult> {
        val shotPixels = bitmapToFloats(shotBitmap)
        val shotHist   = computeHistogram(shotBitmap)

        return doorEntries.map { door ->
            val cos  = cosineSim(shotPixels, door.regionRGB)
            val hist = cosineSim(shotHist, door.histogram)
            val ssim = ssim(shotPixels, door.regionRGB, 450, 750)
            val combined = cos * W_COS + hist * W_HIST + ssim * W_SSIM

            MatchResult(
                filename           = door.filename,
                cosineSimilarity   = cos,
                histogramSimilarity = hist,
                ssimSimilarity     = ssim,
                combinedScore      = combined
            )
        }.sortedByDescending { it.combinedScore }
    }

    /**
     * 用预提取特征向量比对（纯浮点数组，最高效）
     */
    fun matchWithFeatures(
        shotRegionPixels: FloatArray,  // 450*750*3
        shotHistogram: FloatArray,       // 64*3
        doorEntries: List<DoorFeatures>
    ): List<MatchResult> {
        return doorEntries.map { door ->
            val cos  = cosineSim(shotRegionPixels, door.regionRGB)
            val hist = cosineSim(shotHistogram, door.histogram)
            val ssim = ssimPixels(shotRegionPixels, door.regionRGB, 450, 750)
            val combined = cos * W_COS + hist * W_HIST + ssim * W_SSIM

            MatchResult(
                filename           = door.filename,
                cosineSimilarity   = cos,
                histogramSimilarity = hist,
                ssimSimilarity     = ssim,
                combinedScore      = combined
            )
        }.sortedByDescending { it.combinedScore }
    }

    // ── 工具函数 ────────────────────────────────────────────────

    fun extractRegion(bmp: Bitmap, xr1: Float, xr2: Float, yr1: Float, yr2: Float,
                      tw: Int, th: Int): Bitmap {
        val w = bmp.width
        val h = bmp.height
        val x1 = (w * xr1).toInt()
        val x2 = (w * xr2).toInt()
        val y1 = (h * yr1).toInt()
        val y2 = (h * yr2).toInt()
        val raw = Bitmap.createBitmap(bmp, x1, y1, x2 - x1, y2 - y1)
        return Bitmap.createScaledBitmap(raw, tw, th, true).also {
            if (it != raw) raw.recycle()
        }
    }

    fun bitmapToFloats(bmp: Bitmap): FloatArray {
        val w = bmp.width
        val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        val floats = FloatArray(w * h * 3)
        for (i in pixels.indices) {
            floats[i * 3]     = ((pixels[i] shr 16) and 0xFF) / 255f
            floats[i * 3 + 1] = ((pixels[i] shr 8)  and 0xFF) / 255f
            floats[i * 3 + 2] = ((pixels[i])         and 0xFF) / 255f
        }
        return floats
    }

    fun computeHistogram(bmp: Bitmap): FloatArray {
        val bins = HIST_BINS
        val hist = FloatArray(bins * 3)
        val w = bmp.width
        val h = bmp.height
        val pixels = IntArray(w * h)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        for (p in pixels) {
            val r = ((p shr 16) and 0xFF) * bins / 256
            val g = ((p shr 8)  and 0xFF) * bins / 256
            val b = (p         and 0xFF) * bins / 256
            hist[r]++
            hist[bins + g]++
            hist[bins * 2 + b]++
        }
        var norm = 0f
        for (v in hist) norm += v * v
        norm = kotlin.math.sqrt(norm)
        if (norm > 0f) for (i in hist.indices) hist[i] /= norm
        return hist
    }

    fun cosineSim(a: FloatArray, b: FloatArray): Float {
        var dot = 0f
        var na = 0f
        var nb = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            na  += a[i] * a[i]
            nb  += b[i] * b[i]
        }
        val denom = kotlin.math.sqrt(na) * kotlin.math.sqrt(nb)
        return if (denom > 0f) dot / denom else 0f
    }

    /**
     * SSIM — 窗口化，滑动平均
     */
    fun ssim(pixels: FloatArray, other: FloatArray, w: Int, h: Int): Float {
        return ssimPixels(pixels, other, w, h)
    }

    fun ssimPixels(a: FloatArray, b: FloatArray, w: Int, h: Int): Float {
        val C1 = 6.5025f
        val C2 = 58.5225f
        val channels = 3
        var total = 0f

        for (c in 0 until channels) {
            var sumA = 0f; var sumB = 0f
            var sumAB = 0f; var sumA2 = 0f; var sumB2 = 0f
            val n = w * h
            val offset = c

            for (i in 0 until n) {
                val av = a[i * channels + offset]
                val bv = b[i * channels + offset]
                sumA  += av
                sumB  += bv
                sumA2 += av * av
                sumB2 += bv * bv
                sumAB += av * bv
            }

            val muA = sumA / n
            val muB = sumB / n
            val sigmaA = sumA2 / n - muA * muA
            val sigmaB = sumB2 / n - muB * muB
            val sigmaAB = sumAB / n - muA * muB

            val num   = (2f * muA * muB + C1) * (2f * sigmaAB + C2)
            val denom = (muA * muA + muB * muB + C1) * (sigmaA + sigmaB + C2)
            total += if (denom > 1e-10f) num / denom else 0f
        }

        return total / channels
    }
}
