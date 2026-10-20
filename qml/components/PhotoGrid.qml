pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import "../theme" as Room36Theme

GridView {
    id: root
    objectName: "photoGrid"
    property var photoModel: null
    property var favoritesController: null
    property string viewSize: "Medium"
    property bool openOnClick: false
    property bool selectOnImageClick: false
    property string emptyText: "No photos in this library"
    signal photoActivated(string photoId)
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
        property bool favoriteValue: root.favoritesController
            ? root.favoritesController.isFavorite(photoId) : false
        property bool favoritePending: root.favoritesController
            ? root.favoritesController.isPending(photoId) : false
        function refreshFavorite() {
            favoriteValue = root.favoritesController
                ? root.favoritesController.isFavorite(photoId) : false
            favoritePending = root.favoritesController
                ? root.favoritesController.isPending(photoId) : false
        }
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
        onPhotoIdChanged: { scheduleAcquire(); refreshFavorite() }
        Component.onCompleted: { mounted = true; scheduleAcquire() }
        Component.onDestruction: { mounted = false; release() }
        Connections {
            target: root
            function onPhotoModelChanged() { delegateRoot.scheduleAcquire() }
            function onThumbnailEdgeChanged() { delegateRoot.scheduleAcquire() }
            function onFavoritesControllerChanged() { delegateRoot.refreshFavorite() }
        }
        Connections {
            target: root.favoritesController
            function onFavoriteChanged(path, favorite) {
                if (path === delegateRoot.photoId) delegateRoot.favoriteValue = favorite
            }
            function onPendingChanged(path, pending) {
                if (path === delegateRoot.photoId) delegateRoot.favoritePending = pending
            }
            function onChanged() { delegateRoot.refreshFavorite() }
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
            favorite: delegateRoot.favoriteValue
            favoriteBusy: delegateRoot.favoritePending
            showFavorite: root.favoritesController !== null && root.favoritesController.loaded
            onFavoriteToggled: function(value) {
                if (root.favoritesController) root.favoritesController.setFavorite(delegateRoot.photoId, value)
            }
            showMenu: false
            cacheImage: false // The service owns the bounded cache, not QML's URL cache.
            onSelectionToggled: function(value) {
                if (root.photoModel) root.photoModel.setSelected(delegateRoot.photoId, value)
            }
            onClicked: {
                if (root.openOnClick) root.photoActivated(delegateRoot.photoId)
                else if (root.photoModel) root.photoModel.setSelected(delegateRoot.photoId, !delegateRoot.selected)
            }
            onImageClicked: {
                if (root.selectOnImageClick || !root.openOnClick) {
                    if (root.photoModel) root.photoModel.setSelected(delegateRoot.photoId, !delegateRoot.selected)
                } else root.photoActivated(delegateRoot.photoId)
            }
            onFooterClicked: {
                if (root.openOnClick) root.photoActivated(delegateRoot.photoId)
                else if (root.photoModel) root.photoModel.setSelected(delegateRoot.photoId, !delegateRoot.selected)
            }
            Accessible.name: delegateRoot.filename + ", " + delegateRoot.badgeText
                + (root.openOnClick && delegateRoot.badgeText.indexOf(" · ") < 0 ? ", undecided" : "")
                + (delegateRoot.thumbnailState === "error" ? ", preview unavailable" : "")
        }
    }

    Text {
        anchors.centerIn: parent
        visible: root.count === 0
        text: root.emptyText
        color: theme.mutedText
        font.family: typography.bodyFamily
        font.pixelSize: typography.body
    }
}
