package com.doormatcher.native

import android.graphics.Bitmap
import android.graphics.ColorSpace
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicBlur

/**
 * 预提取的门图特征向量（用于快速比对）
 */
class DoorFeatures(
    val filename: String,
    val fullWidth: Int,
    val fullHeight: Int,
    val regionRGB: FloatArray,   // 450*750*3 归一化浮点
    val histogram: FloatArray,   // 64*3 归一化直方图
    val fullMean: FloatArray,    // 全图 RGB 均值
    val fullVar: FloatArray      // 全图 RGB 方差
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is DoorFeatures) return false
        return filename == other.filename
    }

    override fun hashCode(): Int = filename.hashCode()
}

/**
 * 纯 Kotlin 实现的三指标相似度比对
 * weightCosine=0.35, weightHistogram=0.15, weightSSIM=0.50
 */
object DoorMatcherNative {

    private const val HIST_BINS = 64
    private const val W_COS  = 0.35f
    private const val W_HIST = 0.15f
    private const val W_SSIM = 0.50f

    data class MatchResult(
        val filename: String,
        val cosine: Float,
        val histogram: Float,
        val ssim: Float,
        val combined: Float
    )

    /**
     * 从 Bitmap 提取区域并缩放到固定尺寸，返回 FloatArray
     */
    fun extractRegion(bitmap: Bitmap, xr1: Float, xr2: Float,
                      yr1: Float, yr2: Float, tw: Int, th: Int): FloatArray {
        val w = bitmap.width
        val h = bitmap.height
        val x1 = (w * xr1).toInt()
        val x2 = (w * xr2).toInt()
        val y1 = (h * yr1).toInt()
        val y2 = (h * yr2).toInt()

        val raw = Bitmap.createBitmap(bitmap, x1, y1, x2 - x1, y2 - y1)
        val scaled = Bitmap.createScaledBitmap(raw, tw, th, true)
        if (scaled != raw) raw.recycle()

        val pixels = IntArray(tw * th)
        scaled.getPixels(pixels, 0, tw, 0, 0, tw, th)
        scaled.recycle()

        val floats = FloatArray(tw * th * 3)
        for (i in pixels.indices) {
            floats[i * 3]     = ((pixels[i] shr 16) and 0xFF) / 255f
            floats[i * 3 + 1] = ((pixels[i] shr 8)  and 0xFF) / 255f
            floats[i * 3 + 2] = ((pixels[i])         and 0xFF) / 255f
        }
        return floats
    }

    /**
     * 计算颜色直方图（64bins × 3通道，L2归一化）
     */
    fun computeHistogram(pixels: FloatArray, w: Int, h: Int): FloatArray {
        val bins = HIST_BINS
        val hist = FloatArray(bins * 3)

        for (i in 0 until w * h) {
            val r = (pixels[i * 3]     * bins).toInt().coerceIn(0, bins - 1)
            val g = (pixels[i * 3 + 1] * bins).toInt().coerceIn(0, bins - 1)
            val b = (pixels[i * 3 + 2] * bins).toInt().coerceIn(0, bins - 1)
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
        return if (denom > 1e-10f) dot / denom else 0f
    }

    /**
     * 全图 SSIM（简化版，无滑动窗口）
     */
    fun ssimPixels(a: FloatArray, b: FloatArray, w: Int, h: Int): Float {
        val C1 = 6.5025f
        val C2 = 58.5225f
        val n = w * h
        var total = 0f

        for (c in 0..2) {
            var sumA = 0f; var sumB = 0f
            var sumA2 = 0f; var sumB2 = 0f; var sumAB = 0f

            for (i in 0 until n) {
                val av = a[i * 3 + c]
                val bv = b[i * 3 + c]
                sumA  += av;  sumB  += bv
                sumA2 += av*av; sumB2 += bv*bv
                sumAB += av*bv
            }

            val muA = sumA / n
            val muB = sumB / n
            val sigmaA  = sumA2 / n - muA * muA
            val sigmaB  = sumB2 / n - muB * muB
            val sigmaAB = sumAB / n - muA * muB

            val num   = (2f * muA * muB + C1) * (2f * sigmaAB + C2)
            val denom = (muA * muA + muB * muB + C1) * (sigmaA + sigmaB + C2)
            total += if (denom > 1e-10f) num / denom else 0f
        }

        return total / 3f
    }

    /**
     * 比对截图区域与门图特征列表，返回排序结果
     */
    fun match(
        shotRegionPixels: FloatArray,
        shotHistogram: FloatArray,
        doorFeatures: List<DoorFeatures>
    ): List<MatchResult> {
        return doorFeatures.map { door ->
            val cos  = cosineSim(shotRegionPixels, door.regionRGB)
            val hist = cosineSim(shotHistogram, door.histogram)
            val ssim = ssimPixels(shotRegionPixels, door.regionRGB, 450, 750)
            val combined = cos * W_COS + hist * W_HIST + ssim * W_SSIM

            MatchResult(
                filename = door.filename,
                cosine   = cos,
                histogram = hist,
                ssim     = ssim,
                combined = combined
            )
        }.sortedByDescending { it.combined }
    }
}
