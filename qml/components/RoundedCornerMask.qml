import QtQuick

Canvas {
    id: root

    property color maskColor: "white"
    property real cornerRadius: 0

    enabled: false
    antialiasing: true
    visible: cornerRadius > 0

    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()
    onMaskColorChanged: requestPaint()
    onCornerRadiusChanged: requestPaint()

    onPaint: {
        const context = getContext("2d")
        const w = width
        const h = height
        const r = Math.max(0, Math.min(cornerRadius, w / 2, h / 2))

        context.clearRect(0, 0, w, h)
        if (r <= 0) {
            return
        }

        context.fillStyle = maskColor

        context.beginPath()
        context.moveTo(0, 0)
        context.lineTo(r, 0)
        context.arc(r, r, r, -Math.PI / 2, -Math.PI, true)
        context.closePath()
        context.fill()

        context.beginPath()
        context.moveTo(w, 0)
        context.lineTo(w - r, 0)
        context.arc(w - r, r, r, -Math.PI / 2, 0, false)
        context.closePath()
        context.fill()

        context.beginPath()
        context.moveTo(w, h)
        context.lineTo(w, h - r)
        context.arc(w - r, h - r, r, 0, Math.PI / 2, false)
        context.closePath()
        context.fill()

        context.beginPath()
        context.moveTo(0, h)
        context.lineTo(r, h)
        context.arc(r, h - r, r, Math.PI / 2, Math.PI, false)
        context.closePath()
        context.fill()
    }
}
