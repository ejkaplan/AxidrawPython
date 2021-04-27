def merge_paths(paths):
    print(f"BEFORE MERGE: {len(paths)-1} pen lifts")
    frontier = paths.copy()
    finished = []
    while len(frontier) > 0:
        curr = frontier.pop(0)
        changed = False
        for other in frontier:
            new_path = None
            if curr[0] == other[0]:
                new_path = curr[:0:-1] + other
            elif curr[0] == other[-1]:
                new_path = other[:-1] + curr
            elif curr[-1] == other[0]:
                new_path = curr[:-1] + other
            elif curr[-1] == other[-1]:
                new_path = curr + other[-2::-1]
            if new_path is not None:
                frontier.remove(other)
                frontier.append(new_path)
                changed = True
                break
        if not changed:
            finished.append(curr)
    print(f"AFTER MERGE: {len(finished)-1} pen lifts")
    return finished
