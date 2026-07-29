package com.doormatcher.ui

import android.os.Bundle
import android.widget.SeekBar
import androidx.appcompat.app.AppCompatActivity
import com.doormatcher.databinding.ActivitySettingsBinding

class SettingsActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySettingsBinding

    companion object {
        const val PREFS_NAME = "door_matcher_prefs"
        const val KEY_WEIGHT_COS  = "weight_cosine"
        const val KEY_WEIGHT_HIST  = "weight_histogram"
        const val KEY_WEIGHT_SSIM  = "weight_ssim"
        const val KEY_REGION_X1    = "region_x1"
        const val KEY_REGION_X2    = "region_x2"
        const val KEY_REGION_Y1    = "region_y1"
        const val KEY_REGION_Y2    = "region_y2"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        loadSettings()
        setupListeners()
    }

    private fun loadSettings() {
        val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)

        binding.seekbarCosine.progress  = (prefs.getFloat(KEY_WEIGHT_COS,  0.35f) * 100).toInt()
        binding.seekbarHistogram.progress = (prefs.getFloat(KEY_WEIGHT_HIST, 0.15f) * 100).toInt()
        binding.seekbarSsim.progress     = (prefs.getFloat(KEY_WEIGHT_SSIM, 0.50f) * 100).toInt()
        binding.seekbarX1.progress       = (prefs.getFloat(KEY_REGION_X1,  0.25f) * 100).toInt()
        binding.seekbarX2.progress      = (prefs.getFloat(KEY_REGION_X2,  0.75f) * 100).toInt()
        binding.seekbarY1.progress      = (prefs.getFloat(KEY_REGION_Y1,  0.25f) * 100).toInt()
        binding.seekbarY2.progress      = (prefs.getFloat(KEY_REGION_Y2,  0.75f) * 100).toInt()

        updateLabels()
    }

    private fun updateLabels() {
        binding.tvCosineValue.text  = "${binding.seekbarCosine.progress}%"
        binding.tvHistogramValue.text = "${binding.seekbarHistogram.progress}%"
        binding.tvSsimValue.text     = "${binding.seekbarSsim.progress}%"
        binding.tvX1Value.text       = "${binding.seekbarX1.progress}%"
        binding.tvX2Value.text       = "${binding.seekbarX2.progress}%"
        binding.tvY1Value.text       = "${binding.seekbarY1.progress}%"
        binding.tvY2Value.text       = "${binding.seekbarY2.progress}%"
    }

    private fun setupListeners() {
        val listener = object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar?, progress: Int, fromUser: Boolean) {
                updateLabels()
            }
            override fun onStartTrackingTouch(sb: SeekBar?) {}
            override fun onStopTrackingTouch(sb: SeekBar?) {}
        }

        binding.seekbarCosine.setOnSeekBarChangeListener(listener)
        binding.seekbarHistogram.setOnSeekBarChangeListener(listener)
        binding.seekbarSsim.setOnSeekBarChangeListener(listener)
        binding.seekbarX1.setOnSeekBarChangeListener(listener)
        binding.seekbarX2.setOnSeekBarChangeListener(listener)
        binding.seekbarY1.setOnSeekBarChangeListener(listener)
        binding.seekbarY2.setOnSeekBarChangeListener(listener)

        binding.btnSave.setOnClickListener { saveSettings() }
        binding.btnReset.setOnClickListener { resetSettings() }
    }

    private fun saveSettings() {
        val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE).edit()
        prefs.putFloat(KEY_WEIGHT_COS,  binding.seekbarCosine.progress  / 100f)
        prefs.putFloat(KEY_WEIGHT_HIST, binding.seekbarHistogram.progress / 100f)
        prefs.putFloat(KEY_WEIGHT_SSIM, binding.seekbarSsim.progress     / 100f)
        prefs.putFloat(KEY_REGION_X1,   binding.seekbarX1.progress       / 100f)
        prefs.putFloat(KEY_REGION_X2,   binding.seekbarX2.progress       / 100f)
        prefs.putFloat(KEY_REGION_Y1,   binding.seekbarY1.progress       / 100f)
        prefs.putFloat(KEY_REGION_Y2,   binding.seekbarY2.progress       / 100f)
        prefs.apply()
        finish()
    }

    private fun resetSettings() {
        binding.seekbarCosine.progress  = 35
        binding.seekbarHistogram.progress = 15
        binding.seekbarSsim.progress     = 50
        binding.seekbarX1.progress       = 25
        binding.seekbarX2.progress       = 75
        binding.seekbarY1.progress       = 25
        binding.seekbarY2.progress       = 75
        updateLabels()
    }
}
