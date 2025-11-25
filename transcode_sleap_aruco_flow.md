```mermaid

graph TD

    subgraph Deigo["Deigo Cluster (CPU & Storage)"]
        Start(Start: .avi Inputs) --> Split["1. Split Job<br/>(ffmpeg segment)"]
        Split --> EncArray["2. Encode Array<br/>(ffmpeg transcode)<br/>[Batched]"]
        EncArray --> EncSync["3. Encode Sync<br/>(Push to /bucket & Mark encode.ok)"]

        EncSync --> ArucoArray["4a. ArUco Array<br/>(run_aruco.py)<br/>[Batched]"]
        EncSync --> Bridge["4b. Bridge Job<br/>(Rsync to Saion & Submit Remote Jobs)"]

        ArucoArray --> ArucoSync["5a. ArUco Sync<br/>(Sync .h5 & Mark aruco.ok)"]
        
        Bridge --> DeigoCleanup["6. Deigo Cleanup<br/>(rm /flash temp files)"]
        ArucoSync --> DeigoCleanup
    end

    subgraph Saion["Saion Cluster (GPU)"]
        Bridge -.-> |SSH Submission| SleapArray["5b. SLEAP Array<br/>(GPU Inference)<br/>[Batched]"]
        SleapArray --> SleapCollect["6. SLEAP Collect<br/>(Rsync results to /bucket)"]
        SleapCollect --> SaionCleanup["7. Saion Cleanup<br/>(rm /work temp files)"]
    end

    subgraph Coordination["Tracking Only"]
        SleapCollect -.-> |Writes| SleapSignal("Sentinel: sleap.ok")
        ArucoSync -.-> |Writes| ArucoSignal("Sentinel: aruco.ok")
    end
```