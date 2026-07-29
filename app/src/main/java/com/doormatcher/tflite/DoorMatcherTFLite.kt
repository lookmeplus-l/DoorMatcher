package com.doormatcher.tflite

import android.content.Context
import android.graphics.Bitmap
import com.doormatcher.native.DoorMatcherNative
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer

/**
 * TFLite 特征匹配方案
 *
 * 训练阶段：从29张门图提取 regionRGB 特征，训练一个轻量分类/嵌入模型
 * 推理阶段：截图片段 → TFLite 嵌入向量 → 与门图嵌入库做余弦相似度检索
 *
 * 模型输入:  [1, 450, 750, 3]  归一化 RGB
 * 模型输出:  [1, 128]          128维嵌入向量
 */
class DoorMatcherTFLite(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var embeddingDim: Int = 128

    /**
     * 加载 TFLite 模型（从 assets 或 filesDir）
     */
    suspend fun load(modelName: String = "door_model.tflite") = withContext(Dispatchers.IO) {
        if (interpreter != null) return@withContext

        try {
            // 优先从 filesDir 加载（可热更新）
            val modelFile = File(context.filesDir, modelName)
            val buffer: MappedByteBuffer = if (modelFile.exists()) {
                FileUtil.loadMappedFile(context, modelFile.absolutePath)
            } else {
                // 回退到 assets
                FileUtil.loadMappedFile(context, "models/$modelName")
            }

            val options = Interpreter.Options().apply {
                numThreads = 4
                useNNAPI = true
            }
            interpreter = Interpreter(buffer, options)
        } catch (e: Exception) {
            e.printStackTrace()
            // 模型不存在时静默降级到纯 Kotlin
        }
    }

    fun unload() {
        interpreter?.close()
        interpreter = null
    }

    val isAvailable: Boolean get() = interpreter != null

    /**
     * 从 Bitmap 提取嵌入向量
     * 返回 FloatArray[embeddingDim]
     */
    fun extractEmbedding(bitmap: Bitmap): FloatArray? {
        val interpreter = interpreter ?: return null

        // 预处理：提取区域并缩放到 450×750
        val xr1 = 0.25f; val xr2 = 0.75f
        val yr1 = 0.25f; val yr2 = 0.75f
        val tw = 450; val th = 750

        val w = bitmap.width
        val h = bitmap.height
        val x1 = (w * xr1).toInt()
        val x2 = (w * xr2).toInt()
        val y1 = (h * yr1).toInt()
        val y2 = (h * yr2).toInt()

        val raw = Bitmap.createBitmap(bitmap, x1, y1, x2 - x1, y2 - y1)
        val scaled = Bitmap.createScaledBitmap(raw, tw, th, true)
        if (scaled != raw) raw.recycle()

        // 转换为 ByteBuffer (NHWC, [0,255] uint8 → 归一化)
        val byteBuffer = ByteBuffer.allocateDirect(1 * tw * th * 3).apply {
            order(ByteOrder.nativeOrder())
        }

        val pixels = IntArray(tw * th)
        scaled.getPixels(pixels, 0, tw, 0, 0, tw, th)
        scaled.recycle()

        for (p in pixels) {
            byteBuffer.putFloat(((p shr 16 and 0xFF) / 255f))
            byteBuffer.putFloat(((p shr 8  and 0xFF) / 255f))
            byteBuffer.putFloat(((p         and 0xFF) / 255f))
        }

        // 推理
        val output = Array(1) { FloatArray(embeddingDim) }
        try {
            interpreter.run(byteBuffer, output)
        } catch (e: Exception) {
            e.printStackTrace()
            return null
        }

        // L2 归一化嵌入向量
        var norm = 0f
        for (v in output[0]) norm += v * v
        norm = kotlin.math.sqrt(norm)
        if (norm > 0f) for (i in output[0].indices) output[0][i] /= norm

        return output[0]
    }

    /**
     * 用嵌入向量与门图数据库比对
     * doorEmbeddings: Map<filename, embedding>
     */
    fun matchWithEmbeddings(
        shotEmbedding: FloatArray,
        doorEmbeddings: Map<String, FloatArray>
    ): List<Pair<String, Float>> {
        return doorEmbeddings.map { (name, emb) ->
            val cos = cosineSim(shotEmbedding, emb)
            name to cos
        }.sortedByDescending { it.second }
    }

    private fun cosineSim(a: FloatArray, b: FloatArray): Float {
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
     * 导出所有门图的嵌入向量到文件（供后续加载）
     */
    suspend fun exportDoorEmbeddings(
        doorBitmaps: Map<String, Bitmap>,
        outputPath: String = "door_embeddings.json"
    ) = withContext(Dispatchers.IO) {
        if (interpreter == null) return@withContext

        val result = StringBuilder()
        result.append("{\n")

        doorBitmaps.entries.forEachIndexed { index, (name, bitmap) ->
            val emb = extractEmbedding(bitmap) ?: return@forEachIndexed
            val values = emb.joinToString(",") { "%.6f".format(it) }
            result.append("  \"$name\": [$values]")
            if (index < doorBitmaps.size - 1) result.append(",")
            result.append("\n")
        }

        result.append("}\n")
        File(context.filesDir, outputPath).writeText(result.toString())
    }
}
