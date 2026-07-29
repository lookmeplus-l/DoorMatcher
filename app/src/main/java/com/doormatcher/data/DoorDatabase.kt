package com.doormatcher.data

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ColorSpace
import android.graphics.Rect
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * 门图数据库
 * 从 assets/doors/ 目录加载所有门图，运行时管理
 */
class DoorDatabase(private val context: Context) {

    // 区域比例配置（x=[25%:75%], y=[25%:75%]）
    private val regionXR1 = 0.25f
    private val regionXR2 = 0.75f
    private val regionYR1 = 0.25f
    private val regionYR2 = 0.75f

    data class DoorEntry(
        val filename: String,
        val fullBitmap: Bitmap,
        val regionBitmap: Bitmap,
        val fullPixels: IntArray,
        val regionPixels: IntArray,
        val fullRGB: FloatArray,
        val regionRGB: FloatArray,
        val histogram: FloatArray
    ) {
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as DoorEntry
            return filename == other.filename
        }

        override fun hashCode(): Int = filename.hashCode()
    }

    private val _doors = mutableListOf<DoorEntry>()
    val doors: List<DoorEntry> get() = _doors

    private val _loaded = java.util.concurrent.atomic.AtomicBoolean(false)
    val isLoaded get() = _loaded.get()

    // 比对参数
    var weightCosine = 0.35f
    var weightHistogram = 0.15f
    var weightSSIM = 0.50f

    /**
     * 异步加载所有门图
     */
    suspend fun load() = withContext(Dispatchers.IO) {
        if (_loaded.get()) return@withContext

        _doors.clear()

        val doorsDir = File(context.filesDir, "doors")
        // 若 filesDir 没有，则从 assets 复制
        if (!doorsDir.exists() || doorsDir.listFiles()?.isEmpty() == true) {
            copyAssetsToFiles()
        }

        doorsDir.listFiles()?.forEach { file ->
            try {
                val opts = BitmapFactory.Options().apply {
                    inPreferredConfig = Bitmap.Config.ARGB_8888
                    inSampleSize = 1
                }
                val full = BitmapFactory.decodeFile(file.absolutePath, opts) ?: return@forEach

                // 截取中央区域
                val rx1 = (full.width  * regionXR1).toInt()
                val rx2 = (full.width  * regionXR2).toInt()
                val ry1 = (full.height * regionYR1).toInt()
                val ry2 = (full.height * regionYR2).toInt()
                val region = Bitmap.createBitmap(full, rx1, ry1, rx2 - rx1, ry2 - ry1)

                // 缩放到统一尺寸 450×750
                val scaledRegion = Bitmap.createScaledBitmap(region, REGION_W, REGION_H, true)

                // 预计算特征向量
                val fullRGB   = bitmapToFeatureRGB(full)
                val regionRGB = bitmapToFeatureRGB(scaledRegion)
                val hist      = computeHistogram(scaledRegion)

                _doors.add(DoorEntry(
                    filename       = file.nameWithoutExtension,
                    fullBitmap     = full,
                    regionBitmap   = scaledRegion,
                    fullPixels     = fullRGB.first,
                    regionPixels   = regionRGB.first,
                    fullRGB        = fullRGB.second,
                    regionRGB      = regionRGB.second,
                    histogram      = hist
                ))

                // 释放临时 bitmap
                if (region != scaledRegion) region.recycle()
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        _loaded.set(true)
    }

    private fun copyAssetsToFiles() {
        try {
            val doorsDir = File(context.filesDir, "doors").also { it.mkdirs() }
            context.assets.list("doors")?.forEach { name ->
                context.assets.open("doors/$name").use { input ->
                    File(doorsDir, name).outputStream().use { output ->
                        input.copyTo(output)
                    }
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun bitmapToFeatureRGB(bmp: Bitmap): Pair<IntArray, FloatArray> {
        val w = bmp.width
        val h = bmp.height
        val pixels = IntArray(w * h)
        val floats = FloatArray(w * h * 3)
        bmp.getPixels(pixels, 0, w, 0, 0, w, h)
        for (i in pixels.indices) {
            floats[i * 3]     = ((pixels[i] shr 16) and 0xFF) / 255f
            floats[i * 3 + 1] = ((pixels[i] shr 8)  and 0xFF) / 255f
            floats[i * 3 + 2] = ((pixels[i])         and 0xFF) / 255f
        }
        return pixels to floats
    }

    private fun computeHistogram(bmp: Bitmap): FloatArray {
        val bins = 64
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
        // L2 归一化
        var norm = 0f
        for (v in hist) norm += v * v
        norm = kotlin.math.sqrt(norm)
        if (norm > 0f) for (i in hist.indices) hist[i] /= norm
        return hist
    }

    companion object {
        const val REGION_W = 450
        const val REGION_H = 750
    }
}
