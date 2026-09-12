# Page strip over left rail

The Pages overview lived in a left rail (`w-56 hidden xl:block`) with a `grid-cols-2` thumbnail grid tuned to `86px` thumbs. On phones that rail either disappeared or stole half the viewport, and the sheet showed raw storage names. We replaced the rail with a single top Page strip that scrolls horizontally with snap, shows outline pills when expanded and `h-28` thumbnails always, and removed every rendered `filename` occurrence while keeping the uuid key for storage and Qdrant.
