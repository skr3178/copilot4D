actions from moving MNIST dataset: 

 1. Camera Shift Comparison (camera_shift.gif)

   Left Side                      Right Side
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Original View (Fixed Camera)   Camera-Shifted View (Ego-Centric)
   Both digits move freely        Camera shifts to keep ego digit centered
   Standard Moving MNIST          Simulates ego-vehicle view

  What you see in the animation:

  • Left: Digits "2" and "7" bounce around as usual
  • Right: The camera shifts to follow digit "2" (ego digit)
  • The action is the camera displacement (UP/DOWN/LEFT/RIGHT)
  • The "7" appears to move more because it's motion + camera shift

  2. Three-Way Comparison (three_way_comparison.gif)

   Left                              Middle                       Right
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Original (Averaged COM)           Ego-Tracked                  Camera-Shifted
   Action = average of both digits   Action = motion of digit 0   Action = camera displacement

  Key Differences

   Aspect          Ego-Tracked (Middle)             Camera-Shifted (Right)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Camera          Fixed                            Moves to follow ego
   Ego digit       Moves in frame                   Stays centered
   Other digit     Moves naturally                  Moves + camera motion
   Action          Ego's displacement               Camera's displacement
   Training task   Predict where other digit goes   Predict world relative to camera

  Why Camera-Shift is Better for CoPilot4D

  Real Autonomous Driving:
  ┌─────────────────────────────────────┐
  │  Camera mounted on ego vehicle      │
  │  → Camera moves WITH the vehicle    │
  │  → Other cars move relative to ego  │
  └─────────────────────────────────────┘

  Camera-Shift MNIST:
  ┌─────────────────────────────────────┐
  │  Camera "mounted" on ego digit      │
  │  → Camera shifts WITH digit 0       │
  │  → Digit 7 moves relative to camera │
  └─────────────────────────────────────┘

  The camera-shift approach more closely matches how CoPilot4D works:

  • Action = ego vehicle pose change (SE(3) transform)
  • Prediction = world state relative to ego
  • Challenge = separate ego motion from world dynamics