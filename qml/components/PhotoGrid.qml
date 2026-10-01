pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import "../theme" as Room36Theme

GridView {
    id: root
    objectName: "photoGrid"
    property var photoModel: null
    property string viewSize: "Medium"
    readonly property int columns: Math.max(1, Math.floor(width / (preferredCardWidth + metrics.photoCardHorizontalGap)))
    readonly property real preferredCardWidth: metrics.photoCardReferenceWidth
        * (viewSize === "Small" ? 0.75 : viewSize === "Large" ? 1.25 : 1)
    readonly property int thumbnailEdge: viewSize === "Large" ? 512 : viewSize === "Small" ? 256 : 384

    Room36Theme.Metrics { id: metrics }
    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }

    model: photoModel
    currentIndex: -1 // Durable selection is independent of GridView's current item.
    cellWidth: Math.max(1, width / columns)
    cellHeight: (cellWidth - metrics.photoCardHorizontalGap) / metrics.photoCardAspectRatio
        + metrics.photoCardFooterHeight + metrics.photoCardHorizontalGap
    cacheBuffer: cellHeight
    reuseItems: false // Destruction releases leases; selection lives in the model.
    clip: true
    boundsBehavior: Flickable.StopAtBounds
    ScrollBar.vertical: ScrollBar {}

    delegate: Item {
        id: delegateRoot
        objectName: "photoGridDelegate"
        required property string photoId
        required property string filename
        required property string thumbnailSource
        required property string thumbnailState
        required property string badgeText
        required property bool selected
        property var leaseModel: null
        property int lease: 0
        property bool mounted: false
        width: root.cellWidth
        height: root.cellHeight

        function release() {
            if (leaseModel && lease) leaseModel.releaseThumbnail(lease)
            lease = 0
            leaseModel = null
        }
        function acquire() {
            if (!mounted) return
            release()
            leaseModel = root.photoModel
            if (leaseModel) lease = leaseModel.acquireThumbnail(photoId, root.thumbnailEdge)
        }
        // GridView can create and discard temporary delegates during a large
        // jump or relayout. Acquire once after that synchronous layout settles.
        function scheduleAcquire() {
            if (mounted) acquisitionTimer.restart()
        }
        Timer {
            id: acquisitionTimer
            interval: 0
            repeat: false
            onTriggered: delegateRoot.acquire()
        }
        onPhotoIdChanged: scheduleAcquire()
        Component.onCompleted: { mounted = true; scheduleAcquire() }
        Component.onDestruction: { mounted = false; release() }
        Connections {
            target: root
            function onPhotoModelChanged() { delegateRoot.scheduleAcquire() }
            function onThumbnailEdgeChanged() { delegateRoot.scheduleAcquire() }
        }

        PhotoCard {
            objectName: "gridPhotoCard"
            anchors.left: parent.left
            anchors.top: parent.top
            width: Math.max(1, parent.width - metrics.photoCardHorizontalGap)
            height: Math.max(1, parent.height - metrics.photoCardHorizontalGap)
            filename: delegateRoot.filename
            source: delegateRoot.thumbnailSource
            selected: delegateRoot.selected
            badgeText: delegateRoot.badgeText
            showFavorite: false // No backend favorite contract exists yet.
            showMenu: false
            cacheImage: false // The service owns the bounded cache, not QML's URL cache.
            onSelectionToggled: function(value) {
                if (root.photoModel) root.photoModel.setSelected(delegateRoot.photoId, value)
            }
            onClicked: {
                if (root.photoModel) root.photoModel.setSelected(delegateRoot.photoId, !delegateRoot.selected)
            }
            Accessible.name: delegateRoot.filename + (delegateRoot.thumbnailState === "error" ? ", preview unavailable" : "")
        }
    }

    Text {
        anchors.centerIn: parent
        visible: root.count === 0
        text: "No photos in this library"
        color: theme.mutedText
        font.family: typography.bodyFamily
        font.pixelSize: typography.body
    }
}
