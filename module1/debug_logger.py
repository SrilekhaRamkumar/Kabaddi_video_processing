# debug_logger.py

def log_gallery(frame_idx, gallery):
    if frame_idx % 30 != 0:
        return

    print("\n" + "█"*70)
    print(f" LOG FOR FRAME {frame_idx:05d} | TOTAL MEMORY ENTRIES: {len(gallery)}")
    print("█"*70)

    for pid, data in gallery.items():
        state = data["kf"].statePost.flatten()
        color_sample = data["feat"][:5]
        m_pos = data["display_pos"] if data["display_pos"] else (0.0, 0.0)
        bb = data["last_bbox"]
        w, h = (bb[2]-bb[0]), (bb[3]-bb[1])
        x = data['age']

        print(f"PLAYER ID: {pid:02d} | Status: {'[VISIBLE]' if data['age']==0 else f'[LOST (Age:{x})]'}")
        print(f" ├─ PHYSICS: Screen_Pos({state[0]:.0f}, {state[1]:.0f}) | Velocity({state[2]:.2f}, {state[3]:.2f})")
        print(f" ├─ COURT:   Width: {m_pos[0]:.2f}m, Depth: {m_pos[1]:.2f}m")
        print(f" ├─ COLORS:  First 5 of 512 bins: {color_sample}")
        print(f" └─ VISUAL:  BBox Height: {h}px | BBox Width: {w}px")
        print("-"*70)
