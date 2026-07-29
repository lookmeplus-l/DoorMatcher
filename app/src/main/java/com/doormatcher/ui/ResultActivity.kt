package com.doormatcher.ui

import android.graphics.Bitmap
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.lifecycleScope
import com.doormatcher.data.DoorDatabase
import com.doormatcher.databinding.ActivityResultBinding
import com.doormatcher.native.DoorMatcherNative
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * 显示比对结果页面
 * 展示 TOP 候选门图及分数
 */
class ResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultBinding
    private lateinit var db: DoorDatabase

    private var screenshotBitmap: Bitmap? = null

    companion object {
        const val EXTRA_SCREENSHOT = "extra_screenshot"
        const val EXTRA_TOP_RESULTS = "extra_top_results"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        hideSystemUI()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        db = DoorDatabase(this)

        @Suppress("DEPRECATION")
        val resultsJson = intent.getStringExtra(EXTRA_TOP_RESULTS) ?: ""

        binding.btnBack.setOnClickListener { finish() }

        displayResults()
    }

    private fun hideSystemUI() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowInsetsControllerCompat(window, window.decorView).let {
            it.hide(WindowInsetsCompat.Type.systemBars())
            it.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }

    private fun displayResults() {
        lifecycleScope.launch {
            db.load()
            withContext(Dispatchers.Main) {
                val results = getTopResults()
                if (results.isEmpty()) {
                    binding.tvBestMatch.text = "未找到匹配结果"
                    return@withContext
                }

                val best = results.first()
                binding.tvBestMatch.text = "🎯 最佳匹配: ${best.filename}\n" +
                        "综合分数: ${"%.1f".format(best.combined * 100)}%\n" +
                        "Cosine: ${"%.1f".format(best.cosine * 100)}%  " +
                        "Histogram: ${"%.1f".format(best.histogram * 100)}%  " +
                        "SSIM: ${"%.1f".format(best.ssim * 100)}%"

                // 加载最佳匹配图片
                loadBestMatchImage(best.filename)
            }
        }
    }

    private suspend fun loadBestMatchImage(filename: String) {
        withContext(Dispatchers.IO) {
            val door = db.doors.find { it.filename == filename }
            if (door != null) {
                withContext(Dispatchers.Main) {
                    binding.ivBestMatch.setImageBitmap(door.fullBitmap)
                }
            }
        }
    }

    private fun getTopResults(): List<DoorMatcherNative.MatchResult> {
        // 从 intent 解析结果，demo 中直接用数据库比对
        if (!db.isLoaded) return emptyList()

        val screenshot = screenshotBitmap
        if (screenshot == null) return emptyList()

        // 提取特征
        val shotRegion = DoorMatcherNative.extractRegion(screenshot, 0.25f, 0.75f, 0.25f, 0.75f, 450, 750)
        val shotHist = DoorMatcherNative.computeHistogram(shotRegion, 450, 750)

        // 构造 DoorFeatures
        val features = db.doors.map { door ->
            DoorMatcherNative.DoorFeatures(
                filename   = door.filename,
                fullWidth  = door.fullBitmap.width,
                fullHeight = door.fullBitmap.height,
                regionRGB  = door.regionRGB,
                histogram  = door.histogram,
                fullMean   = floatArrayOf(),
                fullVar    = floatArrayOf()
            )
        }

        return DoorMatcherNative.match(shotRegion, shotHist, features)
    }

    override fun onDestroy() {
        super.onDestroy()
        screenshotBitmap?.recycle()
    }
}
