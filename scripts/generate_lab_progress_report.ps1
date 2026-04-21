param(
    [string]$OutputPath = "C:\Users\Makoto Hiroi\Downloads\AntsArray_Lab_Progress_Report_2026-04-09.pptx"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function RgbColor([int]$r, [int]$g, [int]$b) {
    return ($r -bor ($g -shl 8) -bor ($b -shl 16))
}

function Set-TextStyle {
    param(
        $TextRange,
        [string]$FontName = "Aptos",
        [int]$Size = 20,
        [int]$Color = 0,
        [switch]$Bold
    )
    $TextRange.Font.Name = $FontName
    $TextRange.Font.Size = $Size
    $TextRange.Font.Color.RGB = $Color
    $TextRange.Font.Bold = [int][bool]$Bold
}

function Add-Textbox {
    param(
        $Slide,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [string]$Text,
        [int]$FontSize = 20,
        [int]$Color = 0,
        [switch]$Bold,
        [int]$Margin = 8
    )

    $shape = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.MarginLeft = $Margin
    $shape.TextFrame.MarginRight = $Margin
    $shape.TextFrame.MarginTop = $Margin
    $shape.TextFrame.MarginBottom = $Margin
    $shape.TextFrame.WordWrap = -1
    Set-TextStyle -TextRange $shape.TextFrame.TextRange -Size $FontSize -Color $Color -Bold:$Bold
    return $shape
}

function Add-Bullets {
    param(
        $Slide,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [string[]]$Items,
        [int]$FontSize = 19,
        [int]$Color = 0
    )

    $shape = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.MarginLeft = 6
    $shape.TextFrame.MarginRight = 6
    $shape.TextFrame.MarginTop = 6
    $shape.TextFrame.MarginBottom = 6

    $shape.TextFrame.TextRange.Text = ($Items -join "`r")
    Set-TextStyle -TextRange $shape.TextFrame.TextRange -Size $FontSize -Color $Color

    for ($i = 1; $i -le $Items.Count; $i++) {
        $paragraph = $shape.TextFrame.TextRange.Paragraphs($i)
        $paragraph.ParagraphFormat.Bullet.Visible = -1
        $paragraph.ParagraphFormat.Bullet.Character = 8226
        $paragraph.ParagraphFormat.SpaceAfter = 4
    }

    return $shape
}

function Add-TitleBlock {
    param(
        $Slide,
        [string]$Title,
        [string]$Subtitle = ""
    )

    $navy = RgbColor 18 40 64
    $sand = RgbColor 240 238 233
    $slate = RgbColor 83 97 114

    $bar = $Slide.Shapes.AddShape(1, 0, 0, 960, 62)
    $bar.Fill.ForeColor.RGB = $navy
    $bar.Line.Visible = 0

    $titleShape = Add-Textbox -Slide $Slide -Left 22 -Top 8 -Width 640 -Height 34 -Text $Title -FontSize 28 -Color $sand -Bold
    $titleShape.TextFrame.MarginTop = 0
    $titleShape.TextFrame.MarginBottom = 0

    if ($Subtitle) {
        $subShape = Add-Textbox -Slide $Slide -Left 24 -Top 68 -Width 680 -Height 30 -Text $Subtitle -FontSize 13 -Color $slate
        $subShape.TextFrame.MarginTop = 0
        $subShape.TextFrame.MarginBottom = 0
    }

    $accent = $Slide.Shapes.AddShape(1, 720, 0, 240, 62)
    $accent.Fill.ForeColor.RGB = RgbColor 208 137 74
    $accent.Line.Visible = 0
    $stamp = Add-Textbox -Slide $Slide -Left 744 -Top 15 -Width 190 -Height 24 -Text "Lab Progress Report" -FontSize 18 -Color (RgbColor 255 255 255) -Bold
    $stamp.TextFrame.TextRange.ParagraphFormat.Alignment = 2
}

function Add-Image {
    param(
        $Slide,
        [string]$Path,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height
    )
    if (-not (Test-Path $Path)) {
        throw "Image not found: $Path"
    }
    return $Slide.Shapes.AddPicture($Path, 0, -1, $Left, $Top, $Width, $Height)
}

function Add-Callout {
    param(
        $Slide,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [string]$Title,
        [string]$Body,
        [int]$FillColor
    )

    $shape = $Slide.Shapes.AddShape(1, $Left, $Top, $Width, $Height)
    $shape.Fill.ForeColor.RGB = $FillColor
    $shape.Line.ForeColor.RGB = RgbColor 255 255 255
    $shape.Line.Weight = 1.25
    $shape.Line.Transparency = 0.2

    $titleShape = Add-Textbox -Slide $Slide -Left ($Left + 10) -Top ($Top + 8) -Width ($Width - 20) -Height 24 -Text $Title -FontSize 16 -Color (RgbColor 255 255 255) -Bold
    $bodyShape = Add-Textbox -Slide $Slide -Left ($Left + 10) -Top ($Top + 28) -Width ($Width - 20) -Height ($Height - 34) -Text $Body -FontSize 13 -Color (RgbColor 255 255 255)
    $bodyShape.TextFrame.WordWrap = -1
}

function Add-DataTable {
    param(
        $Slide,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [string[]]$Headers,
        [object[][]]$Rows,
        [int]$HeaderFill,
        [int]$BodyFill = 0xFFFFFF,
        [int]$HeaderFont = 12,
        [int]$BodyFont = 11
    )

    $tableShape = $Slide.Shapes.AddTable($Rows.Count + 1, $Headers.Count, $Left, $Top, $Width, $Height)
    $table = $tableShape.Table

    for ($c = 1; $c -le $Headers.Count; $c++) {
        $cell = $table.Cell(1, $c)
        $cell.Shape.Fill.ForeColor.RGB = $HeaderFill
        $cell.Shape.TextFrame.TextRange.Text = [string]$Headers[$c - 1]
        Set-TextStyle -TextRange $cell.Shape.TextFrame.TextRange -Size $HeaderFont -Color (RgbColor 255 255 255) -Bold
    }

    for ($r = 0; $r -lt $Rows.Count; $r++) {
        for ($c = 0; $c -lt $Headers.Count; $c++) {
            $cell = $table.Cell($r + 2, $c + 1)
            $cell.Shape.Fill.ForeColor.RGB = $BodyFill
            $cell.Shape.TextFrame.TextRange.Text = [string]$Rows[$r][$c]
            Set-TextStyle -TextRange $cell.Shape.TextFrame.TextRange -Size $BodyFont -Color (RgbColor 35 45 60)
        }
    }

    return $tableShape
}

function Add-ExperimentCascade {
    param(
        $Slide,
        [int[]]$Highlighted = @(),
        [int]$HighlightFill = 0
    )

    $labels = @(
        "Baseline",
        "Tag size",
        "Tuning",
        "Failure",
        "NN trials",
        "Frame rescue",
        "Temporal fill",
        "Code-space"
    )

    $left = 28
    $top = 102
    $totalWidth = 904
    $gap = 7
    $height = 28
    $count = $labels.Count
    $blockWidth = ($totalWidth - ($gap * ($count - 1))) / $count

    $mutedFill = RgbColor 227 231 236
    $mutedText = RgbColor 76 89 105
    $lightText = RgbColor 255 255 255

    for ($i = 0; $i -lt $count; $i++) {
        $x = $left + ($i * ($blockWidth + $gap))
        $isHighlighted = $Highlighted -contains ($i + 1)

        $shape = $Slide.Shapes.AddShape(1, $x, $top, $blockWidth, $height)
        $shape.Line.Visible = 0
        $shape.Fill.ForeColor.RGB = $(if ($isHighlighted) { $HighlightFill } else { $mutedFill })

        $shape.TextFrame.TextRange.Text = $labels[$i]
        $shape.TextFrame.MarginLeft = 2
        $shape.TextFrame.MarginRight = 2
        $shape.TextFrame.MarginTop = 4
        $shape.TextFrame.MarginBottom = 2
        $shape.TextFrame.TextRange.ParagraphFormat.Alignment = 2
        Set-TextStyle -TextRange $shape.TextFrame.TextRange -Size 10 -Color $(if ($isHighlighted) { $lightText } else { $mutedText }) -Bold:$isHighlighted

        if ($i -lt ($count - 1)) {
            $arrow = $Slide.Shapes.AddTextbox(1, $x + $blockWidth, $top + 1, $gap, $height)
            $arrow.TextFrame.TextRange.Text = ">"
            $arrow.TextFrame.TextRange.ParagraphFormat.Alignment = 2
            Set-TextStyle -TextRange $arrow.TextFrame.TextRange -Size 11 -Color (RgbColor 150 158 168) -Bold
            $arrow.Line.Visible = 0
            $arrow.Fill.Visible = 0
        }
    }
}

$repoRoot = Split-Path -Parent $PSScriptRoot

$imgSizePlot = Join-Path $repoRoot "benchmark\size_comparison_plot.png"
$imgTimeseries = Join-Path $repoRoot "benchmark\timeseries_detection.png"
$imgVisual = Join-Path $repoRoot "benchmark\visual_inspection_cam12.png"
$imgBenchmarkAll = Join-Path $repoRoot "nn-aruco-detection-test\results\benchmark_comparison_all.png"
$imgDictFollowup = Join-Path $repoRoot "benchmark\results\size_comparison_visualization.png"
$imgDeepAruco = Join-Path $repoRoot "nn-aruco-detection-test\results\deeparuco_pt_progression.png"

$outputDir = Split-Path -Parent $OutputPath
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

$ppt = $null
$presentation = $null

try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $ppt.Visible = -1
    $presentation = $ppt.Presentations.Add()

    $presentation.PageSetup.SlideWidth = 960
    $presentation.PageSetup.SlideHeight = 540

    $bg = RgbColor 248 246 242
    $navy = RgbColor 18 40 64
    $slate = RgbColor 83 97 114
    $teal = RgbColor 45 122 131
    $amber = RgbColor 208 137 74
    $mint = RgbColor 96 158 125
    $rose = RgbColor 173 91 91

    $slideIndex = 0
    function New-BlankSlide {
        param([ref]$IndexRef, $Presentation, [int]$Background)
        $IndexRef.Value++
        $slide = $Presentation.Slides.Add($IndexRef.Value, 12)
        $slide.FollowMasterBackground = 0
        $slide.Background.Fill.ForeColor.RGB = $Background
        return $slide
    }

    # Slide 1
    $slide = New-BlankSlide -IndexRef ([ref]$slideIndex) -Presentation $presentation -Background $bg
    $hero = $slide.Shapes.AddShape(1, 0, 0, 960, 540)
    $hero.Fill.ForeColor.RGB = RgbColor 245 242 236
    $hero.Line.Visible = 0
    $banner = $slide.Shapes.AddShape(1, 0, 0, 960, 150)
    $banner.Fill.ForeColor.RGB = $navy
    $banner.Line.Visible = 0
    Add-Textbox -Slide $slide -Left 34 -Top 32 -Width 620 -Height 60 -Text "Ant-Mounted ArUco ID Recovery" -FontSize 30 -Color (RgbColor 255 255 255) -Bold | Out-Null
    Add-Textbox -Slide $slide -Left 34 -Top 82 -Width 700 -Height 40 -Text "Benchmarking, NN rescue attempts, and temporal gap fill" -FontSize 20 -Color (RgbColor 224 230 237) | Out-Null
    Add-Callout -Slide $slide -Left 40 -Top 188 -Width 275 -Height 160 -Title "Project aim" -Body "Track individual ants reliably across long recordings. Improve ID coverage while avoiding false positives that corrupt downstream tracking." -FillColor $teal
    Add-Callout -Slide $slide -Left 336 -Top 188 -Width 275 -Height 160 -Title "Core question" -Body "When OpenCV misses a frame, is the missing information recoverable by tuning, learned models, crop rescue, or only by using time?" -FillColor $amber
    Add-Callout -Slide $slide -Left 632 -Top 188 -Width 288 -Height 160 -Title "Current answer" -Body "Best production path so far: OpenCV full-frame detection plus conservative temporal gap fill on SLEAP-linked tracklets." -FillColor $mint
    Add-Textbox -Slide $slide -Left 42 -Top 390 -Width 500 -Height 22 -Text "Lab progress report • 2026-04-09 • Branch: feature/nn-aruco-detection" -FontSize 15 -Color $slate | Out-Null
    Add-Textbox -Slide $slide -Left 42 -Top 430 -Width 840 -Height 56 -Text "Data sources used in this deck: benchmark/protocol.md, nn-aruco-detection-test/protocol.md, benchmark plots, and the latest CSV/log outputs in nn-aruco-detection-test/results/." -FontSize 18 -Color (RgbColor 35 45 60) | Out-Null

    # Slide 2
    $slide = New-BlankSlide -IndexRef ([ref]$slideIndex) -Presentation $presentation -Background $bg
    Add-TitleBlock -Slide $slide -Title "Background and Experimental Setup" -Subtitle "Why this work matters and what was benchmarked"
    Add-Bullets -Slide $slide -Left 34 -Top 110 -Width 430 -Height 300 -FontSize 20 -Color (RgbColor 35 45 60) -Items @(
        "OpenCV ArUco is already strong, but misses clustered frames when ants move near arena walls or present steep tag tilt.",
        "False positives are costly because wrong IDs propagate into downstream tracking and behavioral analysis.",
        "Two complementary datasets were used: 19 single-ant cameras for fair ground truth, and 4 dense-nest cameras for realistic multi-ant stress tests.",
        "We tested physical changes, detector tuning, learned detectors, crop-rescue strategies, and finally temporal recovery."
    ) | Out-Null
    Add-DataTable -Slide $slide -Left 500 -Top 120 -Width 400 -Height 160 -Headers @("Dataset", "Purpose", "Scale") -Rows @(
        @("Single-ant 19-cam set", "Fair GT and tag-size tests", "4024x3036, 24 fps"),
        @("Dense nest 4-cam set", "Multi-ant realism and FP stress", "~60 ants / camera"),
        @("SLEAP + ArUco chunk data", "Temporal propagation benchmark", "Tracklet-level fill")
    ) -HeaderFill $navy -BodyFill (RgbColor 255 255 255) -HeaderFont 13 -BodyFont 12 | Out-Null
    Add-Callout -Slide $slide -Left 500 -Top 310 -Width 190 -Height 110 -Title "Baseline" -Body "OpenCV full-frame detection: 87-100% in the single-ant study before temporal recovery." -FillColor $teal
    Add-Callout -Slide $slide -Left 710 -Top 310 -Width 190 -Height 110 -Title "Target" -Body "Recover the remaining misses without increasing wrong IDs." -FillColor $amber

    # Slide 3
    $slide = New-BlankSlide -IndexRef ([ref]$slideIndex) -Presentation $presentation -Background $bg
    Add-TitleBlock -Slide $slide -Title "Trial Roadmap" -Subtitle "Purpose of each trial / benchmark and the verdict so far"
    Add-DataTable -Slide $slide -Left 24 -Top 96 -Width 912 -Height 390 -Headers @("Trial", "Purpose", "Main test / benchmark", "Verdict so far") -Rows @(
        @("Tag size", "Check whether larger markers materially improve detection", "run_size_comparison.py on 19 single-ant videos", "All sizes workable; 2.0 mm is the best compromise"),
        @("OpenCV parameter sweep", "See if tuning rescues missed frames", "aruco_benchmark.py on 6 cameras", "No true gain; low perimeter rate only adds FPs"),
        @("Failure analysis", "Explain why misses remain", "time series, sharpness, visual inspection", "Geometry / tilt near walls dominates"),
        @("YOLO + classifier", "Use NN for localization and ID classification", "benchmark_real.py", "Location OK, ID precision poor"),
        @("YOLO + OpenCV / cascade", "Localize with YOLO, decode with OpenCV", "single-ant + dense nest benchmarks", "Useful on hard cases, but still below OpenCV on fair GT"),
        @("DeepArUco / RT-DETR", "Try learned corner/bit decoding pipelines", "sample-frame and real benchmarks", "Did not beat classical pipeline"),
        @("Whitelist / SLEAP crop rescue", "Recover frame-local undecoded markers", "whitelist_experiment.py + rescue ablations", "Too little yield or mostly noise"),
        @("Temporal gap fill", "Recover unreadable frames using time", "benchmark_temporal.py", "Best remaining gain at ~99.9% precision"),
        @("Dictionary-space follow-up", "Reduce ghost IDs by safer code-space", "d1000 vs d250/d100/d50 + subset search", "Promising direction for future filming")
    ) -HeaderFill $navy -BodyFill (RgbColor 255 255 255) -HeaderFont 12 -BodyFont 10 | Out-Null

    # Slide 4
    $slide = New-BlankSlide -IndexRef ([ref]$slideIndex) -Presentation $presentation -Background $bg
    Add-TitleBlock -Slide $slide -Title "Track A: Tag Size Benchmark" -Subtitle "Purpose: determine whether physical marker size, not code-space, is the main performance limiter"
    Add-ExperimentCascade -Slide $slide -Highlighted @(2) -HighlightFill $teal
    Add-Bullets -Slide $slide -Left 34 -Top 144 -Width 340 -Height 220 -FontSize 19 -Color (RgbColor 35 45 60) -Items @(
        "19 single-ant cameras, IDs 3 / 17 / 25, sizes 1.5 / 2.0 / 2.5 mm.",
        "Mean detection rate by size: 1.5 mm = 94.5%, 2.0 mm = 95.0%, 2.5 mm = 97.5%.",
        "Between-camera variation was larger than the 3-point gap between smallest and largest tags.",
        "Conclusion: size matters modestly; 2.0 mm is a practical compromise."
    ) | Out-Null
    Add-Callout -Slide $slide -Left 34 -Top 374 -Width 340 -Height 88 -Title "Why this trial mattered" -Body "If size had been the dominant bottleneck, we could have solved the problem physically and avoided extra pipeline complexity." -FillColor $teal
    Add-Image -Slide $slide -Path $imgSizePlot -Left 410 -Top 144 -Width 500 -Height 318 | Out-Null

    # Slide 5
    $slide = New-BlankSlide -IndexRef ([ref]$slideIndex) -Presentation $presentation -Background $bg
    Add-TitleBlock -Slide $slide -Title "Track B: Parameter Sweep and Failure Analysis" -Subtitle "Purpose: test whether OpenCV tuning can recover true misses"
    Add-ExperimentCascade -Slide $slide -Highlighted @(3,4) -HighlightFill $amber
    Add-Bullets -Slide $slide -Left 30 -Top 144 -Width 300 -Height 254 -FontSize 18 -Color (RgbColor 35 45 60) -Items @(
        "The dramatic boost from lowering minMarkerPerimeterRate was misleading: it only added false positives.",
        "CLAHE and APRILTAG refinement reduced performance on full-frame detection.",
        "Misses cluster in 5-30 s bands when ants stay near arena walls.",
        "Sharpness and brightness were unchanged between hit and miss frames; perspective distortion was the key failure mode.",
        "ID 25 is more fragile than ID 17 because the bit pattern is harder to resolve under tilt."
    ) | Out-Null
    Add-Image -Slide $slide -Path $imgTimeseries -Left 360 -Top 144 -Width 265 -Height 170 | Out-Null
    Add-Image -Slide $slide -Path $imgVisual -Left 640 -Top 144 -Width 270 -Height 170 | Out-Null
    Add-Callout -Slide $slide -Left 360 -Top 336 -Width 550 -Height 108 -Title "Decision from this stage" -Body "Keep baseline OpenCV parameters for full-frame detection. The remaining misses are not recoverable by simple detector tuning; they are mostly geometry-driven." -FillColor $amber

    # Slide 6
    $slide = New-BlankSlide -IndexRef ([ref]$slideIndex) -Presentation $presentation -Background $bg
    Add-TitleBlock -Slide $slide -Title "NN Detector Trials" -Subtitle "Purpose: test whether learned localization / decoding can beat OpenCV on hard frames"
    Add-ExperimentCascade -Slide $slide -Highlighted @(5) -HighlightFill $rose
    Add-Bullets -Slide $slide -Left 32 -Top 140 -Width 305 -Height 258 -FontSize 18 -Color (RgbColor 35 45 60) -Items @(
        "YOLO + classifier improved localization but produced many wrong IDs.",
        "YOLO + OpenCV hybrid removed most classifier mistakes but still lost to OpenCV on fair single-ant GT.",
        "YOLO + cascade gave modest gains on hard cameras only.",
        "DeepArUco-PT suffered a strong sim-to-real gap in decoder quality.",
        "Bottom line: no frame-level NN variant became the production winner."
    ) | Out-Null
    Add-Callout -Slide $slide -Left 32 -Top 410 -Width 305 -Height 62 -Title "Key fair benchmark result" -Body "OpenCV still won camera-by-camera when the metric was correct-ID detection rate." -FillColor $rose
    Add-Image -Slide $slide -Path $imgBenchmarkAll -Left 360 -Top 142 -Width 550 -Height 230 | Out-Null
    Add-Image -Slide $slide -Path $imgDeepAruco -Left 360 -Top 386 -Width 265 -Height 86 | Out-Null
    Add-Callout -Slide $slide -Left 644 -Top 386 -Width 266 -Height 86 -Title "Interpretation" -Body "The learned models often found marker-like regions, but not the correct ID under real distortions." -FillColor $teal

    # Slide 7
    $slide = New-BlankSlide -IndexRef ([ref]$slideIndex) -Presentation $presentation -Background $bg
    Add-TitleBlock -Slide $slide -Title "Frame-Level Rescue Ablations" -Subtitle "Purpose: exhaust single-frame recovery options before shifting to temporal reasoning"
    Add-ExperimentCascade -Slide $slide -Highlighted @(6) -HighlightFill $amber
    Add-DataTable -Slide $slide -Left 34 -Top 144 -Width 560 -Height 198 -Headers @("Rescue strategy", "Purpose", "Result") -Rows @(
        @("YOLO + cascade", "Try multiple crop paddings / preprocessors / decode profiles", "Modest +5 to +6 pp on hard cameras"),
        @("Whitelist matcher", "Accept only session-valid IDs from decoded quads", "35 / 1685 accepted (2.1%), 0 wrong IDs"),
        @("SLEAP crop rescue", "Decode from SLEAP-centered crops when OpenCV misses", "Best arm decoded 897 / 6674 (13.4%)"),
        @("Cross-check", "Test whether those decodes were real", "0 / 897 track-consistent, 0 / 897 YOLO agreement")
    ) -HeaderFill $navy -BodyFill (RgbColor 255 255 255) -HeaderFont 12 -BodyFont 11 | Out-Null
    Add-Bullets -Slide $slide -Left 620 -Top 152 -Width 280 -Height 198 -FontSize 19 -Color (RgbColor 35 45 60) -Items @(
        "Most frame-local rescue detections were noise, not recovered truth.",
        "The unresolved misses are usually face-down, occluded, or too distorted to decode in a single frame.",
        "This shifted the strategy from better frame decoding to using time."
    ) | Out-Null
    Add-Callout -Slide $slide -Left 620 -Top 366 -Width 280 -Height 96 -Title "Take-home message" -Body "Single-frame rescue is largely exhausted. The remaining information has to come from temporal continuity, not another crop-level trick." -FillColor $amber

    # Slide 8
    $slide = New-BlankSlide -IndexRef ([ref]$slideIndex) -Presentation $presentation -Background $bg
    Add-TitleBlock -Slide $slide -Title "Temporal Gap Fill" -Subtitle "Purpose: recover unreadable frames using track continuity instead of frame-local decoding"
    Add-ExperimentCascade -Slide $slide -Highlighted @(7) -HighlightFill $navy
    Add-Bullets -Slide $slide -Left 34 -Top 144 -Width 290 -Height 185 -FontSize 18 -Color (RgbColor 35 45 60) -Items @(
        "Build SLEAP tracklets by nearest-neighbor linking across frames.",
        "Only fill interior gaps when the same ID appears on both sides.",
        "Never overwrite existing detections; max gap length = 10 frames.",
        "Designed to prefer missing labels over risky guesses."
    ) | Out-Null
    Add-DataTable -Slide $slide -Left 348 -Top 144 -Width 260 -Height 160 -Headers @("Synthetic drop", "Recovery", "Fill precision") -Rows @(
        @("10%", "97.1%", "99.9%"),
        @("20%", "96.8%", "99.9%"),
        @("30%", "96.3%", "99.9%"),
        @("50%", "94.4%", "99.9%")
    ) -HeaderFill $teal -BodyFill (RgbColor 255 255 255) -HeaderFont 12 -BodyFont 12 | Out-Null
    Add-DataTable -Slide $slide -Left 635 -Top 144 -Width 270 -Height 160 -Headers @("Natural gain", "Coverage increase") -Rows @(
        @("cam04", "+6.8%"),
        @("cam05", "+9.1%"),
        @("cam09", "+14.8%"),
        @("cam10", "+9.6%")
    ) -HeaderFill $mint -BodyFill (RgbColor 255 255 255) -HeaderFont 12 -BodyFont 12 | Out-Null
    Add-Callout -Slide $slide -Left 40 -Top 344 -Width 865 -Height 112 -Title "Production recommendation from this stage" -Body "OpenCV full-frame detection plus conservative temporal propagation is the most reliable combined pipeline so far. This is the cleanest remaining gain and preserves very high precision." -FillColor $navy

    # Slide 9
    $slide = New-BlankSlide -IndexRef ([ref]$slideIndex) -Presentation $presentation -Background $bg
    Add-TitleBlock -Slide $slide -Title "2026-04-09 Follow-Up: Dictionary-Space Experiments" -Subtitle "Purpose: reduce ghost IDs and improve robustness by choosing a safer code-space"
    Add-ExperimentCascade -Slide $slide -Highlighted @(8) -HighlightFill $mint
    Add-Bullets -Slide $slide -Left 34 -Top 144 -Width 300 -Height 222 -FontSize 18 -Color (RgbColor 35 45 60) -Items @(
        "Re-ran the single-ant size comparison with 4x4_1000, 4x4_250, and 4x4_50.",
        "Smaller dictionaries increased raw detections, but some cameras went above 100%, showing extra detections too.",
        "Distance search on DICT_4X4_1000 found a max 94-ID subset with rotational Hamming distance 4.",
        "ID census found 14 IDs that appear only in d1000 and disappear in d250 (714 detections total)."
    ) | Out-Null
    Add-Image -Slide $slide -Path $imgDictFollowup -Left 360 -Top 144 -Width 550 -Height 270 | Out-Null
    Add-Callout -Slide $slide -Left 360 -Top 426 -Width 270 -Height 60 -Title "Practical option" -Body "<=250 ants: DICT_4X4_250 is the easiest safer replacement." -FillColor $teal
    Add-Callout -Slide $slide -Left 640 -Top 426 -Width 270 -Height 60 -Title "Higher-separation option" -Body "~100 ants: custom 94-ID subset with min distance 4." -FillColor $amber

    # Slide 10
    $slide = New-BlankSlide -IndexRef ([ref]$slideIndex) -Presentation $presentation -Background $bg
    Add-TitleBlock -Slide $slide -Title "Conclusions So Far" -Subtitle "What we know, what is ready, and what to do next"
    Add-Bullets -Slide $slide -Left 38 -Top 112 -Width 410 -Height 290 -FontSize 20 -Color (RgbColor 35 45 60) -Items @(
        "OpenCV remains the best frame-level detector for correct-ID recovery in this setting.",
        "Tag geometry and code-space matter more than more aggressive image preprocessing.",
        "The remaining frame-level misses are mostly unreadable in a single frame.",
        "Temporal gap fill gives the best additional coverage with near-zero risk.",
        "Dictionary reduction / subset design is the strongest future hardware-side improvement."
    ) | Out-Null
    Add-DataTable -Slide $slide -Left 500 -Top 116 -Width 380 -Height 165 -Headers @("Decision", "Status") -Rows @(
        @("Production path", "OpenCV + temporal gap fill"),
        @("NN detector research", "Archive as R&D, not mainline"),
        @("Future filming", "Prefer d250 or a high-separation subset"),
        @("Immediate next test", "Re-check whether cascade still helps after temporal fill")
    ) -HeaderFill $navy -BodyFill (RgbColor 255 255 255) -HeaderFont 12 -BodyFont 12 | Out-Null
    Add-Callout -Slide $slide -Left 500 -Top 314 -Width 380 -Height 150 -Title "Recommended message for the lab" -Body "We tested the obvious physical, tuning, and learned-detector rescue routes. The best near-term gain is not a new detector, but temporal recovery plus a safer marker dictionary for future experiments." -FillColor $mint

    $presentation.SaveAs($OutputPath)
}
finally {
    if ($presentation -ne $null) {
        $presentation.Close()
    }
    if ($ppt -ne $null) {
        $ppt.Quit()
    }
}

Write-Host "Saved presentation to $OutputPath"
