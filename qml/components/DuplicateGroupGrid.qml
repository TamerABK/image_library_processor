pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import "../theme" as Room36Theme

GridView {
    id: root
    objectName: "duplicateGroupGrid"
    property var groupModel: null
    property string viewSize: "Medium"
    signal groupOpened(string groupId)
    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }

    model: groupModel
    readonly property int scrollbarGutter: 18
    readonly property real preferredWidth: 232 * (viewSize === "Small" ? 0.8 : viewSize === "Large" ? 1.25 : 1)
    readonly property int columns: Math.max(1, Math.floor((width - scrollbarGutter) / preferredWidth))
    cellWidth: Math.max(1, (width - scrollbarGutter) / columns)
    cellHeight: Math.max(190, Math.min(290, cellWidth * 1.0))
    cacheBuffer: cellHeight
    reuseItems: false
    clip: true
    boundsBehavior: Flickable.StopAtBounds
    ScrollBar.vertical: ScrollBar { width: 8 }

    delegate: Item {
        id: tile
        objectName: "duplicateGroupDelegate"
        required property string groupId
        required property int photoCount
        required property int recommendedKeeperCount
        required property string representativePhotoId
        required property bool reviewed
        required property bool hasConflict
        required property string recommendationState
        required property string previewSource
        property var leaseModel: null
        property int lease: 0
        property bool mounted: false
        width: root.cellWidth
        height: root.cellHeight

        function release() {
            if (leaseModel && lease) leaseModel.releasePreview(lease)
            lease = 0
            leaseModel = null
        }
        function acquire() {
            if (!mounted) return
            release()
            leaseModel = root.groupModel
            if (leaseModel) lease = leaseModel.acquirePreview(groupId, 256)
        }
        Timer { id: acquireTimer; interval: 0; onTriggered: tile.acquire() }
        onGroupIdChanged: if (mounted) acquireTimer.restart()
        Component.onCompleted: { mounted = true; acquireTimer.start() }
        Component.onDestruction: { mounted = false; release() }
        Connections {
            target: root
            function onGroupModelChanged() { acquireTimer.restart() }
        }

        Rectangle {
            id: card
            objectName: "duplicateGroupCard"
            anchors.left: parent.left
            anchors.top: parent.top
            width: Math.max(1, parent.width - 12)
            height: Math.max(1, parent.height - 12)
            radius: metrics.photoCardRadius
            color: theme.panelFill
            border.width: cardMouse.containsMouse ? 2 : 1
            border.color: cardMouse.containsMouse ? theme.focusRing : theme.separator
            clip: true
            Rectangle {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                height: parent.height - 68
                color: theme.imagePlaceholder
                Image {
                    anchors.fill: parent
                    source: tile.previewSource
                    asynchronous: true
                    cache: false
                    fillMode: Image.PreserveAspectCrop
                }
            }
            Column {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                anchors.margins: 12
                spacing: 3
                Text {
                    text: tile.photoCount + (tile.photoCount === 1 ? " photo" : " photos")
                    color: theme.mainBlue
                    font.family: typography.buttonFamily
                    font.pixelSize: typography.body
                }
                Text {
                    objectName: "duplicateGroupStatus"
                    text: tile.hasConflict ? "Decision conflict" : tile.reviewed ? "Reviewed" : tile.recommendationState === "ready"
                        ? (tile.recommendedKeeperCount === 1
                            ? "Recommended keeper ready" : "Keeper recommendations ready")
                        : tile.recommendationState === "unavailable" ? "Keeper recommendation unavailable"
                        : "Keeper recommendation pending…"
                    color: theme.mutedText
                    font.family: typography.bodyFamily
                    font.pixelSize: typography.caption
                    elide: Text.ElideRight
                    width: parent.width
                }
            }
            MouseArea {
                id: cardMouse
                objectName: "duplicateGroupOpen"
                anchors.fill: parent
                hoverEnabled: true
                onClicked: root.groupOpened(tile.groupId)
                Accessible.role: Accessible.Button
                Accessible.name: tile.photoCount + " photo duplicate group, "
                    + (tile.hasConflict ? "decision conflict" : tile.reviewed ? "reviewed" : "needs review")
            }
        }
    }
}
